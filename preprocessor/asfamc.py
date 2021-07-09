import numpy as np
from pathlib import Path
import pickle
import os 
from utils_ccy.util.directory import mk_dir
from tqdm import tqdm
import amc_parser as amc

class Processor() :
    def __init__(self, arg) :
        self.arg = arg
        self.raw_data_dir = Path(arg.raw_data_dir)

    def init_values(self) : 
        self.n_data = 0
        self.datas_dict = dict() # 
        self.sample_name = [] # 파일 이름
        self.sample_label = [] # 파일 라벨 
        
    def load_data(self) :  
        '''
        base_dir 안에 있는 c3d 파일들을 (N, C, T, V, M) shape를 가진 하나의 npy 파일로 합친 후, out_path에 저장 

        - output -
        base_dir에 존재하는 skeleton data를 (N, C, T, V, M) shape를 가진 npy 파일로 저장  
        '''

        # 데이터 파일 이름 및 fp 전체 길이 추출  
        self.raw_data_list = list(self.raw_data_dir.glob('subjects/*/*.amc'))
        process = tqdm(self.raw_data_list)
        process.set_description('전체 길이 추출')
        for i, filename in enumerate(process): 
            subject = int(filename.stem.split('_')[0])
            action = int(filename.stem.split('_')[1])
            self.sample_name.append(filename)
            self.sample_label.append(action)  

            data = self.read_xyz(filename) # data: (C, T, V)
            
            # 120 Frame 타노스화
            frame_rate_file = filename.parent.joinpath('frame_rate.txt')
            frame_60 = Path.is_file(frame_rate_file)

            if frame_60 == False : # 120 Frame이면 타노스시켜야함
                data = data[:, 0::2, :]
            
            # Frame의 절반을 겹쳐서 뽑아야 하므로 숫자 중복해서 세기
            
            self.n_data += data.shape[1] // self.arg.max_frame # 0에서 시작
            
            # half frame에서 시작된 개수
            half_num = ((data.shape[1] - int(self.arg.max_frame / 2)) // self.arg.max_frame) 
            if half_num >= 0 : 
                self.n_data += half_num
            
            data = np.expand_dims(data, axis = 0) # data: (N, C, T, V) 
            self.datas_dict[i] = data # 데이터 임시 저장 

            # for debug 
            if i > 3 and self.arg.debug : 
                break

        fp = np.zeros((self.n_data, 3, self.arg.max_frame, self.arg.num_joint), dtype=np.float32) 

        # 데이터 파일 이름
        index = 0
        process = tqdm(self.sample_name)
        for i, filename in enumerate(process) :
            process.set_description('데이터 Load')

            data = self.datas_dict[i] # data: (N, C, T, V) 

            # 0 frame에서 시작 
            for n in range(data.shape[2] // self.arg.max_frame) :
                fp[index] = data[:, :, n*self.arg.max_frame:(n+1)*self.arg.max_frame, :] # fp: (N, C, T, V) 
                index += 1
            
            # Half frame에서 시작 
            for n in range((data.shape[2] - int(self.arg.max_frame / 2)) // self.arg.max_frame) :
                fp[index] = data[:, :, n*self.arg.max_frame + int(self.arg.max_frame/2):(n+1)*self.arg.max_frame + int(self.arg.max_frame/2), :] # fp: (N, C, T, V) 
                index += 1
                
            # for debug 
            if i > 3 and self.arg.debug : 
                break  

        self.fp = np.expand_dims(fp, axis = -1) # M 차원 생성
        
    def pre_processing(self) :  
        self.fp, self.pose_mean, self.pose_max = self.pre_normalization(self.fp)
        
        if self.arg.node_mask != None : 
            self.fp = self.fp[:, :, :, self.arg.node_mask, :]
            self.pose_mean = self.pose_mean[:, :, :, self.arg.node_mask, :]
          
    def save_data(self) :        
        mk_dir(self.arg.save_dir)
        with open('{}/full_joint.pkl'.format(self.arg.save_dir), 'wb') as f : 
            pickle.dump(self.fp_dict, f)
                
    def read_xyz(self, filename) : 
        '''
        skeleton 파일 정보에서 xyz 값만 (C, T, V, M) shape를 가진 numpy 형태로 만들어서 출력 

        - input - 
        filename(pathlib.PosixPath):

        - output -
        data(numpy.array): 
        '''
        
        asf_path = filename.parent / (filename.parent.stem + '.asf')

        joints = amc.parse_asf(asf_path)
        motions = amc.parse_amc(filename)
        numFrame = len(motions)
        data = np.zeros(shape = (3, numFrame, self.arg.num_joint)) # data: (C, T, V)

        # 모든 프레임 데이터 정보 읽기 
        for i in range(numFrame) :
            joints['root'].set_motion(motions[i])
            c_joints = joints['root'].to_dict()

            for j, joint in enumerate(c_joints.values()) : 
                data[:, i, j] = [joint.coordinate[0, 0], joint.coordinate[1, 0], joint.coordinate[2, 0]]

        return data # data: (C, T, V)
    
    def pre_normalization(self, data):

        '''
        - Input -
        data(numpy.array): (N, C, T, V, M) shape를 가진 numpy array.
        examples (N), channels (C), frames (T), nodes (V), persons (M))

        - Output -
        data(numpy.array): (N, C, T, V, M) shape를 가진 numpy array.
        examples (N), channels (C), frames (T), nodes (V), persons (M))
        '''

        # examples (N), channels (C), frames (T), nodes (V), persons (M))
        N, C, T, V, M = data.shape
        s = np.transpose(data, [0, 4, 2, 3, 1])  # to (N, M, T, V, C)

        ################################################# Global Translation ######################################################

        # Joint 데이터를 x, y, z = 0, 0, 0 좌표 근처로 이동
        print('subtract the center joint #1 (spine joint in ntu and neck joint in kinetics)')
        for i_s, skeleton in enumerate(tqdm(s)): # skeleton: (M, T, V, C)
            # Person 데이터가 전부 0인 경우(Person 1과 Person 2가 전부 0)이면 넘기기
            if skeleton.sum() == 0:
                continue

            # Use the first skeleton's body center (`1:2` along the nodes dimension)  
            # 첫번째 Person의 복부 부분의 Joint를 Main body center로 저장(12번 Joint)
            '''
            이상한 게 첫번째 프레임의 2번 Joint 좌표를 중심으로 보는 것이 아니라
            모든 프레임에서 2번 Joint 좌표를 중심으로 바라봄.
            이 경우, 달리기는 제자리기 뛰기가 되는 단점이 존재할 것 같음.
            '''

            main_body_center = skeleton[0][:, 0:1, :].copy() # skeleton: (M, T, V, C) -> main body center: (T, 1, C)

            # 전체 스켈레톤에 대해
            for i_p, person in enumerate(skeleton): # Dimension M (# person)  # person : (T, V, C)

                # Person 데이터가 전부 0인 경우(Person 1과 Person 2가 전부 0)이면 넘기기(continue)
                if person.sum() == 0: # person : (T, V, C)  frames (T), nodes (V), channels (C)
                    continue

                # For all `person`, compute the `mask` which is the non-zero channel dimension
                # Person 데이터가 존재하는 경우, mask는 X, Y, Z 값의 합이 0이 아니면(Action이 있으면) True, 0이면 False
                mask = (person.sum(-1) != 0).reshape(T, V, 1) 

                # Subtract the first skeleton's centre joint
                # Skeleton data의 2번째 Joint를 모두 0, 0, 0으로 이동시킴.
                s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask

        data = np.transpose(s, [0, 4, 2, 3, 1])

        # 데이터 Unit -> Inches -> length로 변환 
        data = data * (1.0 / 0.45) * 2.54 / 100.0
        
        # 데이터 Normalize
        data, pose_mean, pose_max = self.standardizing(data)
        
        return data, pose_mean, pose_max
    

    def standardizing(self, data) :
        pose_mean = np.mean(data, axis = (0, 2), keepdims=True)
        pose_max = np.max(np.abs(data-pose_mean), axis = (0, 2, 3, 4), keepdims=True) # (1, 3, 1, 1, 1)    
        return (data - pose_mean) / pose_max, pose_mean, pose_max    
        

    def start(self) : 
        self.init_values()
        self.load_data()
        self.pre_processing() 
        self.fp_dict = {'data': self.fp, 'pose_mean':self.pose_mean, 'pose_max':self.pose_max}
                
        # 데이터 저장 
        self.save_data()

# if __name__ == "__main__":