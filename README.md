# CRISP
## ComputeR vIsion for Sport Performance 

The goal of the project is to increase sportsmen performances by using Digital technologies. Sportsmen actions are recorded in an ecological situation of their exercises and games.
The first targeted sport is Table Tennis. The goal of automtic analysis is to index video recordings by recognising table tennis strokes in them. An ecological corpus is being recorded by students of Sport Faculty of University of Bordeaux, teachers and table tennis players. The corpus is annotated by experts via crowd-sourced interface. The methodology of action recognition is based on specifically designed Deep Learning architecture and motion analysis. A Fine-grain characterization of actions is then foreseen to optimize performances of sportsmen.
The goal of this repository is to allow research team, PhD students, master students... to be able to reproduce our work, method in the aim to compare our method with theirs or enriched ours. The code is lighter for better understanding.

This work is under the [Creative Commons Attribution 4.0 International license - CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), meaning:

> You are free to:<br>
> Share — copy and redistribute the material in any medium or format<br>
> Adapt — remix, transform, and build upon the material for any purpose, even commercially. 
>    
> Under the following terms:<br>
> Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

 If you use this work, please cite it:

``
Pierre-Etienne Martin, Jenny Benois-Pineau, Renaud Péteri, and Julien Morlier.  Fine grained sport action recognition with twin spatio-temporal convolutional neural networks. Multimedia Tools and Applications, Apr 2020.
``

[bib source](PeMTAP20.bib)

and/or

``
Pierre-Etienne Martin, Jenny Benois-Pineau, Renaud Péteri, and Julien Morlier. Optimal choice of motion estimation methods for fine-grained action classification with 3d convolutional networks. In ICIP 2019, pages 554–558. IEEE, 2019.
``

[bib source](PeICIP19.bib)

and/or

``
Pierre-Etienne Martin, Jenny Benois-Pineau, Renaud Péteri, and Julien Morlier. 3D attention mechanisms in Twin Spatio-Temporal Convolutional Neural Networks. Application to  action classification in videos of table tennis games. In 25th International Conference on Pattern Recognition (ICPR2020) - MiCo Milano Congress Center, Italy, 10-15 January 2021.
``

[bib source](PeICPR20.bib)

# Dataset
## TTStroke-21

The dataset has been annotated using 20 stroke classes and a rejection class.
The dataset contains private data and is available through [MediaEval workshop](https://multimediaeval.github.io/) where we organize the [Sport task](https://multimediaeval.github.io/2020-Sports-Video-Classification-Task/) based on TTStroke-21. To have access to the data, particular conditions need to be accepted. We are working on sharing this dataset while respecting the General Data Protection Regulation (EU GDPR).

# Method
## Twin Spatio-Temporal Concvolutional Neural Network (TSTCNN)

Our network take as input the optical flow computed from the rgb images and the rgb data. The size of the input data is set to (W x H x T) = (120 x 120 x 100).

### Data preparation

From videos we extract the frames, compute the flow (deepflow here) and compute the region of interest from the flow values. We lauch several threads to be faster. The max values of the optical flow data are saved and used later for the normalization process.

```python
############################################################
##################### Build the data #######################
############################################################
def build_data(video_list, save_path, width_OF=320, log=None, workers=15, flow_method='DeepFlow'):
    make_path(save_path)

    # Extract Frames
    extract_frames(video_list, save_path, width_OF, log)

    # Compute DeepFlow
    compute_DeepFlow(video_list, save_path, log, workers)

    # Compute ROI
    compute_ROI(video_list, save_path, log, workers, flow_method=flow_method)


##################### RGB #######################
def extract_frames(video_list, save_path, width_OF, log):
    # Chrono
    start_time = time.time()

    for idx, video_path in enumerate(video_list):
        
        video_name = os.path.basename(video_path)
        progress_bar(idx, len(video_list), 'Frame extraction - %s' % (video_name))

        path_data_video = os.path.join(save_path, video_name.split('.')[0])
        make_path(path_data_video)
        path_RGB = os.path.join(path_data_video, 'RGB')
        make_path(path_RGB)

        # Load Video
        cap = cv2.VideoCapture(video_path)
        length_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_number = 0
        
        # Check if video uploaded
        if not cap.isOpened():
            sys.exit("Unable to open the video, check the path.\n")

        while frame_number < length_video:
            # Load video
            _, rgb = cap.read()

            # Check if load Properly
            if _ == 1:
                # Resizing and Save
                rgb = cv2.resize(rgb, (width_OF, rgb.shape[0] * width_OF // rgb.shape[1]))
                cv2.imwrite(os.path.join(path_RGB, '%08d.png' % frame_number), rgb)
                frame_number += 1
        cap.release()

    progress_bar(idx+1, len(video_list), 'Frame extraction completed in %d s' % (time.time() - start_time), 1, log=log)

##################### Deep Flow #######################
def compute_DeepFlow(video_list, save_path, log, workers):
    start_time = time.time()
    DeepFlow_pool = ActivePool()

    for idx, video_path in enumerate(video_list):

        video_name = os.path.basename(video_path).split('.')[0]
        path_data_video = os.path.join(save_path, video_name)

        # Split the calculation in severals process
        while threading.activeCount() > workers:
            progress_bar(idx + 1 - threading.activeCount(), len(video_list), 'DeepFlow computation')
            time.sleep(0.1)

        if threading.activeCount() <= workers:
            job = threading.Thread(target = compute_DeepFlow_video, name = idx, args = (DeepFlow_pool,
                                                                                        os.path.join(path_data_video, 'RGB'),
                                                                                        os.path.join(path_data_video, 'DeepFlow')))
            job.daemon=True
            job.start()

    while threading.activeCount()>1:
        progress_bar(idx + 1 - threading.activeCount(), len(video_list), 'DeepFlow computation')
        time.sleep(0.1)

    progress_bar(idx + 1, len(video_list), 'DeepFlow computation done in %d s' % (time.time() - start_time), 1, log=log)


def compute_DeepFlow_video(pool, path_RGB, path_Flow):
    name = threading.current_thread().name
    pool.makeActive(name)
    os.system('python deep_flow.py -i %s -o %s' % (path_RGB, path_Flow))
    pool.makeInactive(name)


##################### ROI #######################
def compute_ROI(video_list, save_path, log, workers, flow_method='DeepFlow'):
    start_time = time.time()
    ROI_pool = ActivePool()

    for idx, video_path in enumerate(video_list):

        video_name = os.path.basename(video_path).split('.')[0]
        path_data_video = os.path.join(save_path, video_name)

        # Split the calculation in severals process
        while threading.activeCount() > workers:
            progress_bar(idx + 1 - threading.activeCount(), len(video_list), 'ROI computation for %s' % (flow_method))
            time.sleep(0.1)

        if threading.activeCount() <= workers:
            job = threading.Thread(target = compute_roi_video, name = idx, args = (ROI_pool,
                                                                                   path_data_video,
                                                                                   flow_method))
            job.daemon=True
            job.start()

    while threading.activeCount()>1:
        progress_bar(idx + 1 - threading.activeCount(), len(video_list), 'ROI computation for %s' % (flow_method))
        time.sleep(0.1)

    join_values_flow(video_list, 'values_flow_%s' % flow_method)
    progress_bar(len(video_list), len(video_list), 'ROI computation for %s completed in %d s' % (flow_method, int(time.time() - start_time)), 1, log=log)


def compute_roi_video(pool, path_data_video, flow_method):
    name = threading.current_thread().name
    pool.makeActive(name)
    os.system('python roi_flow.py -v %s -m %s' % (path_data_video, flow_method))
    pool.makeInactive(name)


def join_values_flow(video_list, name_values, save_path):
    values_flow = []
    for video in video_list:
        video_name = os.path.basename(video_path).split('.')[0]
        path_data_video = os.path.join(save_path, video_name)
        values_flow_video = np.load(os.path.join(path_data_video, '%s.npy' % name_values))
        values_flow.extend(values_flow_video)
    np.save(os.path.join(save_path, name_values), values_flow)
```


### Architecture

In our work, we have compared our Twin model with a single branch model. The single branch process either rgb data or optical flow data.

```python
import torch.nn as nn
import torch.nn.functional as F

##########################################################################
########################  Flatten Features  ##############################
##########################################################################
def flatten_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

##########################################################################
####################### Twin RGB OptFlow ##############################
##########################################################################
class NetTwin(nn.Module):
    def __init__(self, size_data, n_classes):
        super(NetTwin, self).__init__()

        ####################
        ####### First ######
        ####################
        self.conv1_RGB = nn.Conv3d(3, 30, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool1_RGB = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

        self.conv1_Flow = nn.Conv3d(2, 30, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool1_Flow = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

        size_data //= 2

        #####################
        ####### Second ######
        #####################
        self.conv2_RGB = nn.Conv3d(30, 60, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool2_RGB = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

        self.conv2_Flow = nn.Conv3d(30, 60, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool2_Flow = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

        size_data //= 2

        ####################
        ####### Third ######
        ####################
        self.conv3_RGB = nn.Conv3d(60, 80, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool3_RGB = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

        self.conv3_Flow = nn.Conv3d(60, 80, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool3_Flow = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

        size_data //= 2

        #####################
        ####### Fusion ######
        #####################
        self.linear_RGB = nn.Linear(80*size_data[0]*size_data[1]*size_data[2], 500)
        self.linear_Flow = nn.Linear(80*size_data[0]*size_data[1]*size_data[2], 500)

        self.linear = nn.Bilinear(500, 500, n_classes)
        self.final = nn.Softmax(1)

    def forward(self, rgb, flow):

        ####################
        ####### First ######
        ####################
        rgb = self.pool1_RGB(F.relu(self.conv1_RGB(rgb)))
        flow = self.pool1_Flow(F.relu(self.conv1_Flow(flow)))

        #####################
        ####### Second ######
        #####################
        rgb = self.pool2_RGB(F.relu(self.conv2_RGB(rgb)))
        flow = self.pool2_Flow(F.relu(self.conv2_Flow(flow)))

        ####################
        ####### Third ######
        ####################
        rgb = self.pool3_RGB(F.relu(self.conv3_RGB(rgb)))
        flow = self.pool3_Flow(F.relu(self.conv3_Flow(flow)))

        #####################
        ####### Fusion ######
        #####################
        rgb = rgb.view(-1, flatten_features(rgb))
        flow = flow.view(-1, flatten_features(flow))

        rgb = F.relu(self.linear_RGB(rgb))
        flow = F.relu(self.linear_Flow(flow))

        data = self.linear(rgb, flow)
        label = self.final(data)

        return label
        
##########################################################################
#############################  One Branch ################################
##########################################################################
class NetSimpleBranch(nn.Module):
    def __init__(self, size_data, n_classes, channels=3):
        super(NetSimpleBranch, self).__init__()

        self.channels = channels
        
        ####################
        ####### First ######
        ####################
        self.conv1 = nn.Conv3d(channels, 30, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)) # dilaion=(0,0,0) (depth, height, width)
        self.pool1 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        size_data //= 2

        ####################
        ###### Second ######
        ####################
        self.conv2 = nn.Conv3d(30, 60, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)) # dilaion=(0,0,0) (depth, height, width)
        self.pool2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        size_data //= 2

        ####################
        ####### Third ######
        ####################
        self.conv3 = nn.Conv3d(60, 80, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)) # dilaion=(0,0,0) (depth, height, width)
        self.pool3 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        size_data //= 2

        ####################
        ####### Last #######
        ####################
        self.linear1 = nn.Linear(80*size_data[0]*size_data[1]*size_data[2], 500) # 144000, 216000
        self.relu = nn.ReLU()

        # Fusion
        self.linear2 = nn.Linear(500, n_classes)
        self.final = nn.Softmax(1)

    def forward(self, rgb, flow):
        if self.channels == 2:
            data = flow
        else:
            data = rgb

        ####################
        ####### First ######
        ####################
        data = self.pool1(F.relu(self.conv1(data))) # data = self.pool1(F.relu(self.drop1(self.conv1(data))))

        ####################
        ###### Second ######
        ####################
        data = self.pool2(F.relu(self.conv2(data)))

        ####################
        ####### Third ######
        ####################
        data = self.pool3(F.relu(self.conv3(data)))

        ####################
        ####### Last #######
        ####################
        data = data.view(-1, flatten_features(data))
        data = self.relu(self.linear1(data))

        data = self.linear2(data)
        label = self.final(data)

        return label
```

### Getting ready
In `args` variable we store all the parameters belonging to the trained models (model_type, augmentation, folders where is saved). We encourage you to modify the code so it fits to your situation. We use the same seed each time before training a model. The data_loader is build from a list of objects which store the name of the video, the temporal segmentation and the type of stroke. Then we create a `Dataset` object able to get the data from one particular sample.


```python
from torch.utils.data import Dataset, DataLoader

#########################################################################
###################### Reset Pytorch Session ############################
#########################################################################
def reset_training(seed):
    gc.collect()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
#################################################################
###################### Model variables ##########################
#################################################################
class my_variables():
    def __init__(self, model_type='Twin', batch_size=10, augmentation=True, nesterov=True, decay=0.005, epochs=500, lr=0.001, momentum=0.5, flow_method='DeepFlow', norm_method='NORMAL', size_data=[100, 120, 120], cuda=True):

        self.model_type = model_type
        self.augmentation = augmentation
        self.nesterov = nesterov
        self.decay = decay
        self.lr = lr
        self.momentum = momentum
        self.flow_method = flow_method
        self.norm_method = norm_method
        self.size_data = np.array(size_data)
        self.model_name = 'pytorch_%s_%s_bs_%s_aug_%d_nest_%d_decay_%s_lr_%s_m_%s_OF_%s_%s_sizeinput_%s_' % (datetime.datetime.now().strftime("%d-%m-%Y_%H-%M"), self.model_type, self.batch_size, self.augmentation, self.nesterov, self.decay, self.lr, self.momentum, self.flow_method, self.norm_method, str(self.size_data))
        self.load = False

        self.epochs = epochs
        self.path_fig_model = os.path.join('Figures', self.model_name)
        make_path(self.path_fig_model)

        if cuda: #Use gpu
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        self.log = setup_logger('model_log', os.path.join(self.path_fig_model, 'log_%s.log' % datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")))

    def state_dict(self):
        dict = self.__dict__.copy()
        del dict['log']
        return dict

##########################################################################
############################ Dataset Class ###############################
##########################################################################
class My_dataset(Dataset):
    def __init__(self, dataset_list, size_data, augmentation=0, norm_method = norm_method, flow_method = flow_method):
        self.dataset_list = dataset_list
        self.augmentation = augmentation
        self.norm_method = norm_method
        self.flow_method = flow_method

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        rgb, flow, label = get_annotation_data(self.dataset_list[idx], self.size_data, augmentation = self.augmentation, norm_method = self.norm_method, flow_method = self.flow_method)
        sample = {'rgb': torch.FloatTensor(rgb), 'flow': torch.FloatTensor(flow), 'label': label}
        return sample
        
#######################################################################
############################ Make model ###############################
#######################################################################
def make_the_model(video_list):

    #### Same seed #####
    reset_training(param.seed)

    ##### Get all the annotations in random order #####
    annotations_list, negative_list = get_annotations_list(video_list)

    ##### Build Train, Validation and Test set #####
    train_list, validation_list, test_list = build_lists_set(annotations_list, negative_list)

    # Variables
    lr = 0.01

    compute_normalization_values(os.path.join(param.path_data, 'values_flow_%s.npy' % 'DeepFlow'))
    
    args = my_variables()

    ##################
    ## Architecture ##
    ##################
    model = make_architecture(args)

    ######################
    ## Data preparation ##
    ######################
    ##### Build Dataset class and Data Loader #####
    train_set = My_dataset(train_list, augmentation = args.augmentation, norm_method = args.norm_method, flow_method = args.flow_method, data_types = args.model_type, fps=args.fps, size_data=args.size_data)
    validation_set = My_dataset(validation_list, norm_method = args.norm_method, flow_method = args.flow_method, data_types = args.model_type, fps=args.fps, size_data=args.size_data)
    test_set = My_dataset(test_list, norm_method = args.norm_method, flow_method = args.flow_method, data_types = args.model_type, fps=args.fps, size_data=args.size_data)

    ## Loaders of the Datasets
    train_loader = DataLoader(train_set, batch_size = args.batch_size, shuffle = True, num_workers = args.workers)
    validation_loader = DataLoader(validation_set, batch_size = args.batch_size, shuffle = False, num_workers = args.workers)
    test_loader = DataLoader(test_set, batch_size = args.batch_size, shuffle = False, num_workers = args.workers)

    ######################
    ## Training process ##
    ######################
    train_model(model, args, train_loader, validation_loader)
    args.load = True

    ## Load best model
    model = make_architecture(args)

    ##################
    ## Test process ##
    ##################
    test_model(model, args, test_loader, param.list_of_moves)
```

### Get the Data, Normalization and Augmentation

Here I provide an example of how to get the data and how to augment them. It is higly dependant on how you saved the data, prepared them, compute. This portion is provided so it can help you in the process of data feeding, normalization and augmentation. In addition, here, the pose is also processed. The get_data function might be called by a Dataset type class which will be used by a dataloader during train, validation and test phases.

```python
###############################################################################
########################### Flow Normalization ################################
###############################################################################
# Get Normalization value #
def compute_normalization_values(path_data, list_videos):
    maxs_x = []
    maxs_y = []
    min_value = 5

    for video_name in list_videos:
        maxs_x_, means_x_, stds_x_, maxs_y_, means_y_, stds_y_ = np.array(list(zip(*np.load(os.path.join(path_data, video_name, 'flow_values_mask.npy')))))
        maxs_x.extend(maxs_x_)
        maxs_y.extend(maxs_y_)
    global mean_x, std_x, mean_y, std_y
    maxs_x = np.array(maxs_x)
    maxs_y = np.array(maxs_y)
    mean_x = maxs_x[maxs_x>min_value].mean()
    std_x = maxs_x[maxs_x>min_value].std()

    mean_y = maxs_y[maxs_y>min_value].mean()
    std_y = maxs_y[maxs_y>min_value].std()
    print('Normalization value:\n \t mean_x: %.2f +- %.2f \n \t mean_y %.2f +- %.2f' %(mean_x, std_x, mean_y, std_y))

def normalize_optical_flow(flow):
    global mean_x, std_x, mean_y, std_y
    # Normal Normalization
    flow_normed = np.dstack((flow[:,:,0] / (mean_x + 3*std_x),
                                flow[:,:,1] / (mean_y + 3*std_y)))
    flow_normed[flow_normed > 1] = 1
    flow_normed[flow_normed < -1] = -1

    return flow_normed

def process_coordinates_pose(coord, zoom, R_matrix, flip, width, height, tx, ty, pose_norm_method='image_size'):
    # y, x
    v = [coord[0],coord[1],1]

    # # Grab  the rotation components of the matrix)
    # cos = np.abs(R_matrix[0, 0])
    # sin = np.abs(R_matrix[0, 1])
    # # compute the new bounding dimensions of the image
    # nW = int((height * sin) + (width * cos))
    # nH = int((height * cos) + (width * sin))
    # # adjust the rotation matrix to take into account translation
    # R_matrix_coord = R_matrix.copy()
    # R_matrix_coord[0, 2] += (nW / 2) - cx
    # R_matrix_coord[1, 2] += (nH / 2) - cy

    # Rotation of the coordinates
    # v = np.dot(R_matrix_coord, v)
    if R_matrix is not None:
        v = np.dot(R_matrix, v)

    new_coord = [zoom*(v[1]+tx), zoom*(v[0]+ty)]

    if flip:
        new_coord[0] = zoom*width - new_coord[0]

    if pose_norm_method=='image_size':
        new_coord = [new_coord[0]/(width*zoom), new_coord[1]/(height*zoom)]

    return new_coord

###############################################################################
################################# Get data ####################################
###############################################################################
def get_data(annotation, size_data, augmentation=0, path_to_save=None, pose_norm_method='image_size', model_type='TwinPose'):
    # Variables
    rgb_data = []
    flow_data = []
    pose_data = []

    crop_Flow = os.path.join(annotation.video_name, 'roi_mask.npy')
    path_RGB = os.path.join(annotation.video_name, 'rgb')
    path_Mask = os.path.join(annotation.video_name, 'mask')
    path_Pose = os.path.join(annotation.video_name, 'pose')
    Pose_parts = ['nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder', 'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist', 'leftHip', 'rightHip']
    path_Flow = os.path.join(annotation.video_name, 'flow')
    x_list, y_list = np.asarray(np.load(crop_Flow)).astype(float)

    rgb_example = cv2.imread(os.path.join(path_RGB, '%08d.png' % 0))
    shape = rgb_example.shape

    if path_to_save is not None:
        # count = len([f for f in os.listdir(path_to_save) if os.path.isfile(os.path.join(path_to_save, f))])/4
        path_to_save = os.path.join(path_to_save, '%04d' % len(os.listdir(path_to_save)))
        make_path(os.path.join(path_to_save))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    # Smoothing of the crop center
    x_list = cv2.GaussianBlur(x_list, (1, int(2 * 1./6 *  120 + 1)), 0)
    y_list = cv2.GaussianBlur(y_list, (1, int(2 * 1./6 *  120 + 1)), 0)
    
    # Random transformations parameters and begin of the interval according to the window
    if augmentation:
        angle, zoom, tx, ty, flip, begin = get_augmentation_parameters(annotation, size_data)
        angle_radian = math.radians(angle)
    else:
        tx = 0
        ty = 0
        flip = False
        zoom = 1
        angle = 0
        begin = (annotation.begin + annotation.end + 1 - size_data[0]) // 2

    begin = max(begin, 0)
    for frame_number in range(begin, begin + size_data[0]):

        x_seg = x_list[frame_number - 1][0]
        y_seg = y_list[frame_number - 1][0]

        if augmentation:
            # Rotation Matrix
            R_matrix = cv2.getRotationMatrix2D((x_seg, y_seg), angle, 1)
        else:
            R_matrix = None

        #################
        ###### Pose #####
        #################
        if 'Pose' in model_type:
            pose = []
            pose_list = np.load(os.path.join(path_Pose, '%08d.npy' % (frame_number)))

            # Take the best pose according to our ROI
            if len(pose_list)==0:
                pose_dict = None
            elif len(pose_list)==1:
                pose_dict = pose_list[0]
            elif len(pose_list) > 1:
                dist = []
                for pose_dict in pose_list:
                    score, coord = pose_dict['Scrore pose']
                    dist.append(sum(abs(coord-np.array([y_seg,x_seg]))))
                pose_dict = pose_list[np.array(dist).argmin()]

            # For each part of the body, keep coordinates and score (default is our ROI)
            if pose_dict is not None:

                score, coord = pose_dict['Scrore pose']
                coord_norm = process_coordinates_pose(coord, zoom, R_matrix, flip, 320, 180, tx, ty, pose_norm_method=pose_norm_method)
                pose.extend([coord_norm[0],coord_norm[1],score])

                for Pose_part in Pose_parts:
                    if flip:
                        if Pose_part[:4] == 'left':
                            score, coord = pose_dict['right'+Pose_part[4:]]
                        elif Pose_part[:5] == 'right':
                            score, coord = pose_dict['left'+Pose_part[5:]]
                        else:
                            score, coord = pose_dict[Pose_part]
                    else:
                        score, coord = pose_dict[Pose_part]
                    pose.extend([coord_norm[0],coord_norm[1],score])

            else:
                coord_norm = process_coordinates_pose((x_seg, y_seg), zoom, R_matrix, flip, 320, 180, tx, ty, pose_norm_method=pose_norm_method)
                for i in range(len(Pose_parts)+1):
                    pose.extend([coord_norm[0],coord_norm[1],0])

            pose_data.append(pose)


        # Update coordinates
        if flip:
            x_seg = shape[1] - x_seg

        # Coordinates correction to fit in the image
        x_seg, y_seg = correction_coordinates(zoom * (x_seg + tx), zoom * (y_seg + ty), size_data[1:], shape)
        x_seg = int(x_seg - size_data[2] * 0.5)
        y_seg = int(y_seg - size_data[1] * 0.5)

        #################
        ###### Flow #####
        #################
        if ('Flow' in model_type) or ('Twin' in model_type):
            try:
                flow = np.load(os.path.join(path_Flow, '%08d.npy' % frame_number))
            except:
                raise ValueError('Problem with %s begin %d inter %d-%d step %d T %d' % (os.path.join(path_Flow, '%08d.npy' % frame_number), begin, annotation.begin, annotation.end, step, size_data[0]))
            
            flow = cv2.GaussianBlur(flow, (3,3), 0)
            mask = cv2.imread(os.path.join(path_Mask, '%08d.png' % frame_number),cv2.IMREAD_GRAYSCALE) / 255
            mask = cv2.dilate(mask, kernel)
            flow = np.multiply(flow, np.dstack((mask, mask)))
            flow = normalize_optical_flow(flow)

            if augmentation:
                flow = apply_augmentation(flow, zoom, R_matrix, angle_radian, flip, flow_values=True)
            
            flow_croped = flow[y_seg : y_seg + size_data[1], x_seg : x_seg + size_data[2]]
            flow_data.append(flow_croped)

        #################
        ###### RGB ######
        #################
        if ('RGB' in model_type) or ('Twin' in model_type):
            try:
                rgb = cv2.imread(os.path.join(path_RGB, '%08d.png' % frame_number)).astype(float) / 255
            except:
                raise ValueError('Problem with %s begin %d inter %d-%d step %d T %d' % (os.path.join(path_RGB, '%08d.png' % frame_number), begin, annotation.begin, annotation.end, step, size_data[0]))

            if augmentation:
                rgb = apply_augmentation(rgb, zoom, R_matrix, angle_radian, flip, flow_values=False)

            rgb_croped = rgb[y_seg : y_seg + size_data[1], x_seg : x_seg + size_data[2]]
            rgb_data.append(cv2.split(rgb_croped))


    label = args.list_of_moves.index(annotation.move)

    if 'Pose' in model_type:
        pose_data = np.transpose(np.array(pose_data), (1, 0))

    if ('RGB' in model_type) or ('Twin' in model_type):
        rgb_data = np.transpose(rgb_data, (1, 0, 2, 3))
    
    if ('Flow' in model_type) or ('Twin' in model_type):
        flow_data = np.transpose(flow_data, (3, 0, 1, 2))
    
    return rgb_data, flow_data, pose_data, label

def correction_coordinates(x, y, size, shape):
    diff = x - size[1] * 0.5
    if diff < 0: x = size[1] * 0.5

    diff = x + size[1] * 0.5 - shape[1]
    if diff > 0: x = shape[1] - size[1] * 0.5

    diff = y - size[0] * 0.5
    if diff < 0: y = size[0] * 0.5

    diff = y + size[0] * 0.5 - shape[0]
    if diff > 0: y = shape[0] - size[0] * 0.5
    return int(x), int(y)

############################ Augmentation ####################################
def get_augmentation_parameters(annotation, size_data, step=1):
    angle = (random.random()* 2 - 1) * 10
    zoom = 1 + (random.random()* 2 - 1) * 0.1

    tx = random.randint(-0.1 * size_data[2], 0.1 * size_data[2])
    ty = random.randint(-0.1 * size_data[1], 0.1 * size_data[1])

    flip = random.randint(0,1)


    # Normal distribution to pick whre to begin #
    mu = annotation.begin + (annotation.end + 1 - annotation.begin - step*size_data[0])/2
    sigma = (annotation.end + 1 - annotation.begin - step*size_data[0])/6
    begin = -1

    if sigma <= 0:
        begin = max(int(mu),0)
    else:
        count=0
        while not annotation.begin <= begin <= annotation.end + 1 - step*size_data[0]:
            begin = int(np.random.normal(mu, sigma))
            count+=1
            if count>10:
                print('Warning: augmentation with picking frame has a problem')

    return angle, zoom, tx, ty, flip, begin


def apply_augmentation(data, zoom, R_matrix, angle_radian, flip, flow_values=False):
    if data is not None:
        # Resize and Rotation
        shape = data.shape
        data = cv2.resize(cv2.warpAffine(data, R_matrix, (shape[1], shape[0])), (0,0), fx = zoom, fy = zoom)
        if flow_values:
            data *= zoom

            # Update Flow values according to rotation
            tmp = cv2.addWeighted(data[:,:,0], math.cos(angle_radian), data[:,:,1], -math.sin(angle_radian), 0)
            data[:,:,1] = cv2.addWeighted(data[:,:,0], math.sin(angle_radian), data[:,:,1], math.cos(angle_radian), 0)
            data[:,:,0] = tmp

        # Flip
        if flip:
            data = cv2.flip(data, 1)
            if flow_values:
                data = -data
    return data

####################################################################
######################### Dataset Class ############################
####################################################################
class My_dataset(Dataset):
    def __init__(self, dataset_list, size_data, augmentation=0, model_type='TwinPose'):
        self.dataset_list = dataset_list
        self.size_data = size_data
        self.augmentation = augmentation
        self.model_type = model_type

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        rgb, flow, pose, label = get_data(self.dataset_list[idx], self.size_data, self.augmentation, model_type = self.model_type)
        sample = {'rgb': torch.FloatTensor(rgb), 'flow' : torch.FloatTensor(flow), 'pose' : torch.FloatTensor(pose), 'label' : label, 'my_stroke' : {'video_name':self.dataset_list[idx].video_name, 'begin':self.dataset_list[idx].begin, 'end':self.dataset_list[idx].end}}
        return sample

class My_test_dataset(Dataset):
    def __init__(self, interval, size_data, augmentation=0, model_type='TwinPose'):
        self.interval = interval
        middle = (interval.begin + interval.end + 1 - size_data[2]) // 2
        n = max(0,(middle - interval.begin))
        self.begin = middle - n
        self.number_of_iteration = n * 2 + 1
        self.size_data = size_data
        self.augmentation = augmentation
        self.model_type = model_type

    def __len__(self):
        return self.number_of_iteration

    def __getitem__(self, idx):
        begin = self.begin + idx
        windowed_interval = MyStroke(self.interval.video_name, begin, begin + self.size_data[2], self.interval.move)
        rgb, flow, pose, label = get_data(windowed_interval, self.size_data, self.augmentation, model_type = self.model_type)
        sample = {'rgb': torch.FloatTensor(rgb), 'flow' : torch.FloatTensor(flow), 'pose' : torch.FloatTensor(pose), 'label': label, 'my_stroke' : {'video_name':self.interval.video_name, 'begin':self.interval.begin, 'end':self.interval.end}}
        return sample

    def my_print(self, show_option=1):
        self.interval.my_print()
        # save_my_dataset(self.annotation, augmentation = self.augmentation, show_option = show_option)
```

### Training Process

During training, we can save and load model. We keep a checkpoint when we perform the best to be able to retrain from this point. We load the data using the dataloader.

```python
from torch.autograd import Variable

##########################################################################
######################## Save and Load Model #############################
##########################################################################
def save_model(model, args, optimizer, epoch, dict_of_values):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dict_of_values': dict_of_values,
                'args': args.state_dict()}, os.path.join(args.path_fig_model, 'model.tar'))

def load_model(model, weigth_path, optimizer=None):
    checkpoint = torch.load(os.path.join(weigth_path, 'model.tar'), map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    dict_of_values = checkpoint['dict_of_values']
    args_dict = checkpoint['args']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return epoch, dict_of_values, args_dict

##########################################################################
########################### Training Process #############################
##########################################################################
def train_epoch(epoch, args, model, data_loader, optimizer, criterion):
    model.train()
    N = len(data_loader.dataset)
    start_time = time.time()
    aLoss = 0
    Acc = 0

    for batch_idx, batch in enumerate(data_loader):
        # Get batch tensor
        rgb, flow, label = batch['rgb'], batch['flow'], batch['label']

        rgb = Variable(rgb.type(args.dtype))
        flow = Variable(flow.type(args.dtype))
        label = Variable(label.type(args.dtype).long())

        optimizer.zero_grad()
        output = model(rgb, flow)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        aLoss += loss.item()
        Acc += output.data.max(1)[1].eq(label.data).cpu().sum().numpy()

    aLoss /= (batch_idx + 1)
    return aLoss, Acc/N


##########################################################################
######################## Validation Process ##############################
##########################################################################
def validation_epoch(epoch, args, model, data_loader, criterion):
    with torch.no_grad():
        N = len(data_loader.dataset)
        aLoss = 0
        Acc = 0

        for batch_idx, batch in enumerate(data_loader):
            # Get batch tensor
            rgb, flow, label = batch['rgb'], batch['flow'], batch['label']

            rgb = Variable(rgb.type(args.dtype))
            flow = Variable(flow.type(args.dtype))
            label = Variable(label.type(args.dtype).long())

            output = model(rgb, flow)

            aLoss += criterion(output, label).item()
            Acc += output.data.max(1)[1].eq(label.data).cpu().sum().numpy()

            progress_bar((batch_idx + 1) * args.batch_size, N, '%d - Validation' % (pid))

        aLoss /= (batch_idx + 1)

        return aLoss, Acc/N

##########################################################################
############################# TRAINING ###################################
##########################################################################
def train_model(model, args, train_loader, validation_loader):
    criterion = nn.CrossEntropyLoss() # change with reduction='sum' -> lr to change
    optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.decay, nesterov = args.nesterov)

    # For plot
    loss_train = []
    loss_val = []
    acc_val = []
    acc_train = []
    max_acc = -1
    acc_val_ = 1
    min_loss_train = 1000
    min_loss_val = 1000

    if args.load:
        print_and_log('Load previous model for retraining', log=args.log)
        epoch, dict_of_values, _ = load_model(model, args.path_fig_model, optimizer=optimizer)
        print_and_log('Model from epoch %d' % (epoch), log=args.log)
        max_acc = dict_of_values['acc_val_']
        min_loss_val = dict_of_values['loss_val_']
        for key in dict_of_values:
            print_and_log('%s : %g' % (key, dict_of_values[key]), log=args.log)
        change_optimizer(optimizer, args, lr=args.lr_max)

    for epoch in range(1, args.epochs+1):

        # Train and validation step and save loss and acc for plot
        loss_train_, acc_train_ = train_epoch(epoch, args, model, train_loader, optimizer, criterion)
        loss_val_, acc_val_ = validation_epoch(epoch, args, model, validation_loader, criterion)

        loss_train.append(loss_train_)
        acc_train.append(acc_train_)
        loss_val.append(loss_val_)
        acc_val.append(acc_val_)

        wait_change_lr += 1

        # Best model saved
        # if (acc_val_ > max_acc) or (acc_val_ >= max_acc and loss_train_ < min_loss_train):
        if min_loss_val > loss_val_:
            save_model(model, args, optimizer=optimizer, epoch=epoch, dict_of_values={'loss_train_': loss_train_, 'acc_train_': acc_train_, 'loss_val_': loss_val_, 'acc_val_': acc_val_})
            max_acc = acc_val_
            min_loss_val = loss_val_
            min_loss_train = loss_train_


    print_and_log('Trained with %d epochs, lr = %g, batchsize = %d, momentum = %g with max validation accuracy of %.2f done in %ds' %\
        (args.epochs, args.lr, args.batch_size, args.momentum, max_acc, time.time() - start_time), log=args.log)

    make_train_figure(loss_train, loss_val, acc_val, acc_train, os.path.join(args.path_fig_model, 'Train.png'))

```


## 3D Attention Mechanism

In this section we provide the code of the attnetion block that we present in "3D attention mechanisms in Twin Spatio-Temporal Convolutional Neural Networks. Application to  action classification in videos of table tennis games.", accepted at ICPR2020. Those blocks can be added to any 3D-CNNs. I modified the implement BatchNorm3d to make it do what I wanted (normalization per channel according to all its values in one sample; otherwise it is done per dimension). Results lead to better performances and faster convergence. Be aware that training parameters might need to be updated.

```python
##########################################################################
############# Batch Normalization 1D for ND tensors  #####################
##########################################################################
class MyBatchNorm(_BatchNorm): ## Replace nn.BatchNorm3d
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(MyBatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        self.saved_shape = input.shape
        if input.dim() != 2 and input.dim() != 3:
            return input.reshape((input.shape[0], input.shape[1], input[0,0].numel()))

    def forward(self, input):
        input = self._check_input_dim(input)
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        output = F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias,
                              self.training or not self.track_running_stats, exponential_average_factor, self.eps)

        output = output.reshape(self.saved_shape)

        return output


###################################################################
####################### 3D Attention Model  #######################
###################################################################
class ResidualBlock3D(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        super(ResidualBlock3D, self).__init__()

        dim_conv = math.ceil(out_dim/4)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stride = stride
        self.bn1 = MyBatchNorm(in_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_dim, dim_conv, 1, 1, bias = False)
        self.bn2 = MyBatchNorm(dim_conv)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(dim_conv, dim_conv, 3, stride, padding = 1, bias = False)
        self.bn3 = MyBatchNorm(dim_conv)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(dim_conv, out_dim, 1, 1, bias = False)
        if (self.in_dim != self.out_dim) or (self.stride !=1 ):
            self.conv4 = nn.Conv3d(in_dim, out_dim , 1, stride, bias = False)

        ## Use GPU
        if param.cuda:
            self.cuda()

    def forward(self, input):
        residual = input
        out = self.bn1(input)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.in_dim != self.out_dim) or (self.stride !=1 ):
            residual = self.conv4(out1)
        out += residual
        return out


class AttentionModule3D(nn.Module):
    def __init__(self, in_dim, out_dim, size1, size2, size3):
        super(AttentionModule3D, self).__init__()

        self.size1 = tuple(size1.astype(int))
        self.size2 = tuple(size2.astype(int))
        self.size3 = tuple(size3.astype(int))

        self.first_residual_blocks = ResidualBlock3D(in_dim, out_dim)

        self.trunk_branches = nn.Sequential(
        	ResidualBlock3D(in_dim, out_dim),
        	ResidualBlock3D(in_dim, out_dim)
        )

        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.block1 = ResidualBlock3D(in_dim, out_dim)

        self.skip1 = ResidualBlock3D(in_dim, out_dim)

        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.block2 = ResidualBlock3D(in_dim, out_dim)

        self.skip2 = ResidualBlock3D(in_dim, out_dim)

        self.pool3 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.block3 = nn.Sequential(
        	ResidualBlock3D(in_dim, out_dim),
        	ResidualBlock3D(in_dim, out_dim)
        )

        self.block4 = ResidualBlock3D(in_dim, out_dim)

        self.block5 = ResidualBlock3D(in_dim, out_dim)

        self.block6 = nn.Sequential(
        	MyBatchNorm(out_dim),
        	nn.ReLU(inplace=True),
        	nn.Conv3d(out_dim, out_dim , kernel_size = 1, stride = 1, bias = False),
        	MyBatchNorm(out_dim),
        	nn.ReLU(inplace=True),
        	nn.Conv3d(out_dim, out_dim , kernel_size = 1, stride = 1, bias = False),
        	nn.Sigmoid()
        )

        self.final = ResidualBlock3D(in_dim, out_dim)

        ## Use GPU
        if param.cuda:
            self.cuda()


    def forward(self, input):
        input = self.first_residual_blocks(input)
        out_trunk = self.trunk_branches(input)

        # 1st level
        out_pool1 =  self.pool1(input)
        out_block1 = self.block1(out_pool1)
        out_skip1 = self.skip1(out_block1)

        #2sd level
        out_pool2 = self.pool2(out_block1)
        out_block2 = self.block2(out_pool2)
        out_skip2 = self.skip2(out_block2)

        # 3rd level
        out_pool3 = self.pool3(out_block2)
        out_block3 = self.block3(out_pool3)
        out_interp3 = F.interpolate(out_block3, size=self.size3, mode='trilinear', align_corners=True)
        out = out_interp3 + out_skip2

        #4th level
        out_softmax4 = self.block4(out)
        out_interp2 = F.interpolate(out_softmax4, size=self.size2, mode='trilinear', align_corners=True)
        out = out_interp2 + out_skip1

        #5th level
        out_block5 = self.block5(out)
        out_interp1 = F.interpolate(out_block5, size=self.size1, mode='trilinear', align_corners=True)

        #6th level
        out_block6 = self.block6(out_interp1)
        out = (1 + out_block6) * out_trunk

        # Final with Attention added
        out_last = self.final(out)

        return out_last
```

## More ?

This should be a good start if you want to train your own models with PyTorch. Of course, you need to implement your onw functions according to your problematic. This code is generic enough and I believe straigh forward enough to be modified and adapted. You can contact me if you meet some difficulties of if you have some correction to make to what it is written so far.

