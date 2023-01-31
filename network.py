import torch.nn as nn

class cnnBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(cnnBlock, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = 1, padding='same'), 
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(),
        )

    def forward(self, x):
        out = self.cnn(x)
        return out 
       
class ResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample = None):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = 1, padding = "same"),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU6())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride = 1, padding = "same"),
                        nn.BatchNorm2d(out_ch))
        self.downsample = downsample
        self.relu = nn.ReLU6()
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

        
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        #encoder     
        self.cnn1 = cnnBlock(1,16)
        self.cnn2 = cnnBlock(16,32)
        self.cnn3 = cnnBlock(32,64)
        self.res = ResnetBlock(64,64) #x5
        # decoder
        self.cnn4 = cnnBlock(64,64)
        self.cnn5 = cnnBlock(64,32)
        self.cnn6 = cnnBlock(32,16)
        self.cnn7 = cnnBlock(16,1)
         

    def forward(self, x):
        r1 = x 
        out1 = self.cnn1(x)
        r2 = out1
        out2 = self.cnn2(out1)
        r3 = out2
        out3 = self.cnn3(out2)
        r4 = out3 

        out4 = self.res(out3)
        out5 = self.res(out4)
        out6 = self.res(out5)
        out7 = self.res(out6)
        out8 = self.relu(self.res(out7) + r4)

        out9 = self.cnn4(out8)
        out10 = self.relu(self.cnn5(out9) + r3)
        out11 = self.relu(self.cnn6(out10) + r2)
        out12 = self.relu(self.cnn7(out11) + r1) 


        return self.sigmoid(out12)