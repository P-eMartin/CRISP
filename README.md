# CRISP
## ComputeR vIsion for Sport Performance 

The goal of the project is to increase sportsmen performances by using Digital technologies. Sportsmen actions are recorded in an ecological situation of their exercises and games.
The first targeted sport is Table Tennis. The goal of automtic analysis is to index video recordings by recognising table tennis strokes in them. An ecological corpus is being recorded by students of Sport Faculty of University of Bordeaux, teachers and table tennis players. The corpus is annotated by experts via crowd-sourced interface. The methodology of action recognition is based on specifically designed Deep Learning architecture and motion analysis. A Fine-grain characterization of actions is then foreseen to optimize performances of sportsmen.
The goal of this repository is to allow research team, PhD students, master students... to be able to reproduce our work, method in the aim to compare our method with theirs or enriched ours. The code is lighter for better understanding.

This work is under the [Creative Commons Attribution 4.0 International license - CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), meaning:

"You are free to:

⋅⋅⋅Share — copy and redistribute the material in any medium or format
    
⋅⋅⋅Adapt — remix, transform, and build upon the material for any purpose, even commercially. 
    
Under the following terms:
⋅⋅⋅Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use."

 If you use this work, please cite it:

``
Pierre-Etienne Martin, Jenny  Benois-Pineau, Renaud  Péteri, and Julien Morlier.  Fine grained sport action recognition with twin spatio-temporal convolutional neural networks. Multimedia Tools and Applications, Apr 2020.
``

[bib source](PeMTAP20.bib)

and/or

``
Pierre-Etienne Martin, Jenny Benois-Pineau, Renaud Péteri, and Julien Morlier. Optimal choice of motion estimation methods  for fine-grained action classification with 3d convolutional networks. In ICIP 2019, pages 554–558. IEEE, 2019.
``

[bib source](PeICIP19.bib)


# Dataset
## TTStroke-21

The dataset has been annotated using 20 stroke classes and a rejection class.
The dataset contains private data and is available through MediaEval workshop where we organize the Sport task based on TTStroke-21. To have access to the data, particular conditions need to be accepted. We are working on sharing this dataset while respecting the General Data Protection Regulation (EU GDPR).

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

    def forward(self, rgb, flowe):

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

## More ?

This should be a good start if you want to train your own models with PyTorch. Of course, you need to implement your onw functions according to your problematic. This code is generic enough and I believe straigh forward enough to be modified and adapted. You can contact me if you meet some difficulties of if you have some correction to make to what it is written so far. Soon we will had supplmentary materials.

