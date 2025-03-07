esc50_path = 'data/esc50'
#esc50_path = 'D:/sound_datasets/esc50'
runs_path = 'results'
# sub-epoch (batch-level) progress bar display
disable_bat_pbar = False#True

# do not change this block
n_classes = 50
folds = 5
test_folds = [1,2,3,4,5]#[1, 2, 3, 4, 5]
# ratio to split off from training data
val_size = .15  # could be changed # 0.2 original # instead 0.25


#model_constructor = "ResNet(block=ResidualBlock, layers=[2, 2, 2, 2], num_classes=config.n_classes)"
#model_constructor = "ResNet(block=ResidualBlock, layers=[3, 4, 6, 3], num_classes=config.n_classes)"
model_constructor = "ResNet_3(BasicBlock_2, [3, 4, 2, 4])"  


# model checkpoints loaded for testing
test_checkpoints =  ['terminal.pt', 'best_val_loss.pt']  #['terminal.pt']
# experiment folder used for testing (result from cross validation training)
#test_experiment = 'results/2024-04-01-00-00'
test_experiment = 'results/results'

# sampling rate for waves
sr = 44100

device_id = 0
batch_size = 32
num_workers = 20 # 4 # for kaggle
persistent_workers = True
epochs = 300           #best so far: 200
patience = 180          #best so far: 50
lr = 1e-3              
weight_decay = 8e-3    #best so far: 8e-3
warm_epochs = 10
gamma = 0.975           #0.94 for 79.8 #0.975 79.6
step_size = 5
