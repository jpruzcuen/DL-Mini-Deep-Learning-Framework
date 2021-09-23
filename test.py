import torch
from torch import empty

import math

torch.set_grad_enabled(False)

#********************** GENERATE DATA **********************************
def generate_disc_set(nb, hot_labels = False):
    input = torch.empty(nb, 2).uniform_(0, 1)
    target = input.sub(0.5).pow(2).sum(1).sub(1 / (2*math.pi)).sign().mul(-1).add(1).div(2).long() 

    if(hot_labels == True):

        target_hot = torch.empty(nb,2).zero_().scatter_(1,target.view(-1,1), 1.0)

        return input, target_hot

    return input, target


#********************** FUN **********************************
# By default activation function "sigma_fct" is Identity() 
# class Fun is used to update activation function. 

class Fun:
    def __init__(x,sigma):
        # if no previous projection layer, just activate input
        if Module.prev_proj == []: 
            temp = sigma
            output = temp(x)
            return output

        else:
            # get previous projection layer
            proj = Module.prev_proj

            # update its activation function
            proj.sigma_fct = sigma

            # update its "activated" feature by applying activation on its output 
            # (forward pass of activation module)
            proj.activated = proj.sigma_fct(proj.output)

        return proj.activated

    # available activation function
    def relu(x):
        return Fun.__init__(x,sigma = ReLU())
  
    def tanh(x):
        return Fun.__init__(x,sigma = Tanh())

    def identity(x):
        return Fun.__init__(x,sigma = Identity())



#********************** MODULE **********************************
class Module:
    # used to pass previous projection layer to activation function
    prev_proj = []

    # list of projection layers, used to compute backward pass (Network or Sequential)
    projs = []

    # list of layers (any kind), used to compute forward pass with Sequential
    layers = []

    # Called when creating a new network (Network or Sequential)
    # reset Module parameters, init loss method feature
    def __init__(self):
        Module.projs = []
        Module.layers = []
        Module.prev_proj = []
        self.loss_mtd = None

    def __call__(self,x):
        return self.forward(x)

    # different in each subclass
    def forward(self):
        pass

    # called from Network or Sequential
    def backward(self):
        # initialization 
        last = self.projs[-1]
        last.dl_dx = self.loss_mtd.dloss(last.activated)
        last.dl_ds = last.sigma_fct.dsigma(last.output) * last.dl_dx

        # iterate through other layers
        start = len(self.projs)-2
        for idx in range(start,-1,-1):
            up = self.projs[idx+1]
            current = self.projs[idx]
            down = self.projs[idx-1]

            current.dl_dx = up.w.t().mm(up.dl_ds.t()).t()
            current.dl_ds = current.sigma_fct.dsigma(current.output) * current.dl_dx 

        #accumulate gradient
        for proj in self.projs:
            proj.dl_dw.add_(proj.dl_ds.t().mm(proj.input)) 
            proj.dl_db = proj.dl_db + proj.dl_ds.t()

    # called from Network or Sequential
    def zeroGrad(self):
        for proj in self.projs:
            proj.dl_dw.zero_()
            proj.dl_db.zero_()

    # called from Network or Sequential
    def step(self, eta):
        for proj in self.projs:
            proj.w = proj.w - eta * proj.dl_dw
            proj.b = proj.b - eta * proj.dl_db

    # callable from any subclass
    def param(self):

        # create a list of tensor [parameters, gradients] for a given projection layer
        def proj_param(self):
            nb_row = self.w.size(0)
            nb_col = self.w.size(1)+1
            prm = torch.empty(nb_row, nb_col)
            grd = torch.empty(nb_row, nb_col)
            prm[:,:-1], prm[:,-1:] = self.w, self.b.view(-1,1)
            grd[:,:-1], grd[:,-1:] = self.dl_dw, self.dl_db.view(-1,1)
            return [prm,grd]
        
        # return [[prm1, grd1],[prm2, grd2],...]
        if ((type(self).__name__ == 'Network') or
            (type(self).__name__ == 'Sequential')):
            temp = []
            for proj in self.projs:
                add = proj_param(proj)
                temp.append(add)
            return temp
            
        # return [parameters, gradients]
        elif (type(self).use == 'projection'):
            return proj_param(self)

        else: return []
        
#********************** LINEAR **********************************
class Linear(Module): 
    use = 'projection'
    def __init__(self,nb_in,nb_out):

        # set during initialization
        self.w = torch.empty(nb_out,nb_in).normal_()
        self.b = torch.empty(nb_out,1).normal_()
        self.dl_dw = torch.empty(nb_out,nb_in)
        self.dl_db = torch.empty(nb_out,1)

        # set during forward pass
        self.input = None
        self.output = None
        self.sigma_fct = None
        self.activated = None 
        
        # set during backward pass
        self.dl_ds = None
        self.dl_dx = None

        # store itself in Module projection list
        Module.projs.append(self)
            

    def forward(self,x):
        # compute forward pass
        self.input = x
        self.output = (self.w.mm(x.t()) + self.b).t()

        # set "by default" activation Module
        self.sigma_fct = Identity()
        self.activated = self.output

        # store itself in Module.prev_proj for later activation (if different)
        Module.prev_proj = self

        return self.output

    
#*********************** ReLU ***********************************
class ReLU(Module):
    use = 'activation'
    def __init__(self):
        # set during forward pass
        self.input = None 
        self.output = None

    def forward(self,x):
        self.input = x
        self.output = self.input.clone()
        self.output[self.output<0] = -1e-3
        return self.output

    def dsigma(self,x):
        temp = x.clone()
        temp[x<0]=-1e-3
        temp[x>=0]=1
        return temp

    # calls its activation procedure, required for Sequential method
    def activate(self,x):
        return Fun.relu(x)
        
#*********************** TANH ***********************************
class Tanh(Module):
    use = 'activation'
    def __init__(self):
        # set during forward pass
        self.input = None 
        self.output = None

    def forward(self,x):
        self.input = x
        self.output = self.input.tanh()
        return self.output

    def dsigma(self,x):
        return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)

    # calls its activation procedure, required for Sequential method
    def activate(self,x):
        return Fun.tanh(x)

#*********************** IDENTITY ***********************************
class Identity(Module):
    use = 'activation'
    def __init__(self):
        # set during forward pass
        self.input = None 
        self.output = None

    def forward(self,x):
        self.input = x
        self.output = self.input
        return self.output

    def dsigma(self,x):
        temp = x.clone()
        return temp.fill_(1.)

    # calls its activation procedure, required for Sequential method
    def activate(x): # no self here, we call it manually (not self called)
        return Fun.identity(x)

    
#*********************** MEAN SQUARE LOSS ***********************************
class MSEloss(Module):
    use = 'computation'
    def __init__(self):
        self.target = None

    def loss(self,output,target):
        self.target = target
        return (output - target).pow(2).sum().item()
    
    def dloss(self,output):
        return 2 * (output - self.target)  
  
    
#*********************** BINARY CROSS ENTROPY LOSS SOFTMAX ***********************************
class BCEloss_soft(Module):
    use = 'computation'
    def __init__(self):
        self.target = None
        self.norm = None

    def loss(self,output,target):
        self.target = target
        self.norm = output.exp().div(output.exp().sum(1).view(-1,1))

        return -(self.target.mul(self.norm.log()) + (1-self.target).mul(1-self.norm).log()).sum().item()
    
    def dloss(self,output):
        return -((self.target.div(self.norm)) - ((1-self.target).div(1-self.norm))) 

    
#*********************** BINARY CROSS ENTROPY LOSS (SIGMOID) ***********************************
class BCEloss_sig(Module):
    use = 'computation'
    def __init__(self):
        self.target = None
        self.norm = None

    def loss(self,output,target):
        self.target = target
        self.norm =  1/(1 + ((-output).exp()))
        return -(self.target.mul(self.norm.log()) + (1-self.target).mul(1-self.norm).log()).sum().item()
    
    def dloss(self,output):
        return -((self.target.div(self.norm)) - ((1-self.target).div(1-self.norm)))


#*********************** NORMAL DEEP NETWORK *********************************** 
class Network(Module):
    def __init__(self):
        # reset Module parameters
        super().__init__()

        # fill self.projs
        self.fc1 = Linear(2,25)
        self.fc2 = Linear(25,25)
        self.fc3 = Linear(25,25)
        self.fc4 = Linear(25,2)

    def forward(self,input):
        x = input
        x = Fun.tanh(self.fc1(x))
        x = Fun.tanh(self.fc2(x))
        x = Fun.tanh(self.fc3(x))
        x = Fun.tanh(self.fc4(x))
        return x

#*********************** SEQUENTIAL ***********************************
class Sequential(Module):
    def __init__(self,*args):
        # reset Module parameters
        super().__init__()

        # init own lists
        self.layers = []
        self.projs = []

        # store any layers in "self.layers"
        for idx,arg in enumerate(args):
            self.layers.append(arg)

            # store projection layers in "self.projs"
            if arg.use == 'projection':
               self.projs.append(arg)

        # number of iteration during forward
        self.end = len(self.layers)-1

    def forward(self,input):
        x = input

        # self.layer index
        idx = 0

        # self.proj index
        j = 0

        # exlore all layers
        while idx <= self.end:
            
            #set current layer
            curent = self.layers[idx]
            curent_type = curent.use

            #set curent input
            if idx == 0: curent.input = x
            else: curent.input = self.layers[idx-1].output

            #set next layer
            if idx < self.end:
                up = self.layers[idx+1]
                up_type = up.use
            else : up = None

            if curent.use == 'projection':

                # projection followed by activation
                if ((up != None) & (up_type == 'activation')):

                    # "activate" function calls Fun.__init__ method:
                    # --> it updates "current.sigma_fct"
                    # --> it computes forward pass of activation layer "up.forward(curent(x))"
                    x = up.activate(curent(x))

                    # update both lists
                    self.layers[idx] = curent
                    self.layers[idx+1] = curent.sigma_fct
                    self.projs[j] = curent 

                    # skip next layer, as already computed
                    idx +=2

                # projection followed by projection
                else : 
                    x = curent(x)
                    self.layers[idx] = curent
                    self.projs[j] = curent
                    idx +=1

                j +=1
                
            else: 
                x = curent(x)
                self.layers[idx] = curent
                idx +=1
        
        return self.layers[-1].output

    
#*********************** SHUFFLE DATA FUNCTION ***********************************
    
def shuffle_data(train_input,train_target,nb_train_samples):
        _,index = torch.empty(nb_train_samples).normal_().sort()
        input = train_input[index]
        target = train_target[index]
        return input,target

#*********************** TRAIN MODEL ***********************************
def train_model(model, train_input, train_target, test_input, test_target, 
                eta = 1e-3, epoch = 35, mini_batch = 100, 
                show_error = False, shuffle = True):

    input,target = train_input,train_target
    nb_train_samples = input.size(0)

    train_data = torch.empty(epoch)
    test_data = torch.empty(epoch)
    
    for e in range(epoch):

        acc_loss = 0
        nb_train_errors = 0

        if shuffle: input, target = shuffle_data(train_input,train_target,nb_train_samples)
        
        for b in range(0, nb_train_samples, mini_batch):

            output = model(input.narrow(0,b,mini_batch))

            sample_loss = model.loss_mtd.loss(output, target.narrow(0,b,mini_batch))
            
            acc_loss = acc_loss + sample_loss
            
            model.zeroGrad()
            model.backward()
            model.step(eta)
            
        train_data[e] = nb_error(model,input,target)
        test_data[e] = nb_error(model,test_input,test_target)

        if(show_error):
            train_error = nb_error(model, input, target) 
            print('epoch: {:d} acc_train_loss: {:.02f} acc_train_error: {:.02f}%'.format(e,acc_loss,(100 * train_error) / nb_train_samples))

    return train_data,test_data

#*********************** COMPUTE NUMBER OF ERRORS ***********************************
def nb_error(model, input, target, mini_batch = 100):

    errors = 0
    nb_train_samples = input.size(0) # 1000

    for b in range(0, nb_train_samples, mini_batch):
            output = model.forward(input.narrow(0,b,mini_batch))
            _, pred = output.max(1) #[1].item() # 100x1
            _, target_pred = target.narrow(0, b, mini_batch).max(1)
            errors += (pred - target_pred).abs().sum() 
    
    return errors


#*********************** MAIN ***********************************

def main():
    
    losses = [["MSELoss",MSEloss()], ["BCELoss (Sigmoid)",BCEloss_sig()], ["BCELoss (Softmax)", BCEloss_soft()]]
    
    eta_both = 5e-4
    epoch_both = 200
    mini_batch_both = 100
    switch = False

    print("\n\nTraining information: eta = {}, epochs = {}, mini_batch = {}".format(eta_both, epoch_both,mini_batch_both))
    print("Training 'Network' and 'Sequential'\n\n")
    
    for index, l in enumerate(losses):
        
        loss_name = l[0]
        loss = l[1]
        
        print("======== Training with {}, {}/{} ========".format(loss_name, index+1, len(losses)))
    
        train_input, train_target = generate_disc_set(1000, hot_labels = True)
        test_input, test_target = generate_disc_set(1000,hot_labels =True)

        mean, std = train_input.mean(), train_input.std()

        train_input.sub_(mean).div_(std)
        test_input.sub_(mean).div_(std)

        mySeq = Sequential(
            Linear(2,25),Tanh(),
            Linear(25,25),Tanh(),
            Linear(25,25),Tanh(),
            Linear(25,2),Tanh()
            )

        mySeq.loss_mtd = loss

        train_data_Seq, test_data_Seq = train_model(mySeq, train_input, train_target,test_input,test_target,
                                                    eta = eta_both, epoch = epoch_both, mini_batch=mini_batch_both,
                                                    show_error=switch)
        
        seq_train_error = 100 * nb_error(mySeq, train_input, train_target, mini_batch_both)/train_input.size(0)
        seq_test_error = 100 * nb_error(mySeq, test_input, test_target, mini_batch_both)/test_input.size(0)

        myNet = Network()
        myNet.loss_mtd = loss

        train_data_Net, test_data_Net =  train_model(myNet, train_input, train_target,test_input,test_target,
                                                    eta = eta_both, epoch = epoch_both, mini_batch=mini_batch_both,
                                                     show_error=switch)
        
        network_train_error = 100 * nb_error(myNet, train_input, train_target, mini_batch_both)/train_input.size(0)
        network_test_error = 100 * nb_error(myNet, test_input, test_target, mini_batch_both)/test_input.size(0)
        
        print("[Sequential] with {}: \nTrain Error = {:.2f}%, Test Error = {:.2f}%".format(loss_name, seq_train_error, seq_test_error))
        print("\n[Network] with {}: \nTrain Error = {:.2f}%, Test Error = {:.2f}%\n\n".format(loss_name, network_train_error, network_test_error))



if __name__ == "__main__":
    main()