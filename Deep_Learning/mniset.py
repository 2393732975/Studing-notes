import torch
import torchvision
from tqdm import tqdm
import matplotlib.pyplot

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.model = torch.nn.Sequential(

            #The size of the picture is 28x28
            torch.nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1),
            #in_channels = 1：输入通道数为1，意味着输入图像是单通道（灰度图）。
            #out_channels = 16：输出通道数为16，卷积层会学习16个不同的特征图。
            #kernel_size = 3：卷积核大小为3x3。
            #stride = 1：步长为1，卷积核每次移动1个单位。
            #padding = 1：在输入图像的边缘添加1个像素的零填充，确保卷积后图像的尺寸不变。
            torch.nn.ReLU(),
            #激活函数ReLU（Rectified Linear Unit），它会将输入中小于0的值变为0，输出中大于0的值保持不变。
            torch.nn.MaxPool2d(kernel_size=2,stride=2),
            #定义一个2x2的最大池化层，步长为2。池化操作会减小图像尺寸并保留最重要的信息（最大值）。
            
            #The size of the picture is 14x14
            torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2),

             #The size of the picture is 7x7
            torch.nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size = 3,stride = 1,padding = 1),
            torch.nn.ReLU(),

            torch.nn.Flatten(),
            #将输入的多维张量展平为一维张量。例如，输入的大小为 (batch_size, 64, 7, 7)，展平后变为 (batch_size, 64 * 7 * 7)。
            torch.nn.Linear(in_features = 7 * 7 * 64, out_features = 128),
            #定义一个全连接层，输入特征数为 7 * 7 * 64 = 3136（上一层的展平结果），输出特征数为128
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128,out_features=10),
            # 定义另一个全连接层，输入特征数为128，输出特征数为10。这里的10表示10个类别，适用于分类任务。
            torch.nn.Softmax(dim=1)
            #使用Softmax函数将输出转换为概率分布，dim=1 表示对每个样本的所有类别进行归一化。Softmax的输出将是一个长度为10的向量，表示每个类别的预测概率。
        )
    def forward(self,input):
        #定义了模型的前向传播过程。前向传播是将输入数据通过各个层进行计算并得到输出。
        output = self.model(input)
        return output





#如果网络能在GPU中训练，就使用GPU；否则使用CPU进行训练
device = "cuda:0" if torch.cuda.is_available() else "cpu"

#这个函数包括了两个操作：将图片转换为张量，以及将图片进行归一化处理
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean = [0.5],std = [0.5])])

path = './data/'

# 下载训练集和测试集
trainData = torchvision.datasets.MNIST(path,train=True,transform=transform,download=True)
testData = torchvision.datasets.MNIST(path,train=True,transform=transform)

# Pytorch中提供了一种叫做DataLoader的方法来让我们进行训练，该方法自动将数据集打包成为迭代器，能够让我们很方便地进行后续的训练处理
BATCH_SIZE = 256

# 构建测试集和数据集的DataLoader
trainDataLoader = torch.utils.data.DataLoader(dataset=trainData,batch_size=BATCH_SIZE,shuffle=True)
testDataLoader = torch.utils.data.DataLoader(dataset=testData,batch_size=BATCH_SIZE)


net =Net()
print(net.to(device))

lossF = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

history = {'Test Loss':[],'Test Accuracy':[]}
EPOCHS = 2
#构建训练循环
for epoch in range(1,EPOCHS+1):
    #构建tqdm进度条
    processBar = tqdm(trainDataLoader,unit='step')
    #打开网络的训练模式
    net.train(True)
    #开始对训练集的DataLoader进行迭代
    for step,(trainImgs,labels) in enumerate(processBar):
        #将图像和标签传输进device中
        trainImgs = trainImgs.to(device)
        labels = labels.to(device)

        #清空模型的梯度
        net.zero_grad()

        #对模型进行前向推理
        outputs = net(trainImgs)

        #计算本轮的loss
        loss = lossF(outputs,labels)
        #计算本轮推理的准确率
        predictions = torch.argmax(outputs,dim=1)
        accuracy = torch.sum(predictions==labels)/labels.shape[0]

        #反向传播
        loss.backward()

        #使用迭代器更新模型权重
        optimizer.step()

        #将本step结果进行可视化处理
        processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" % 
                                   (epoch,EPOCHS,loss.item(),accuracy.item()))
        if step == len(processBar)-1:
            #构造临时变量
            correct,totalLoss = 0,0
            #关闭模型的训练状态
            net.train(False)
            #对测试集的DataLoader进行迭代

            with torch.no_grad():
                for testImgs,labels in testDataLoader:
                    testImgs = testImgs.to(device)
                    labels = labels.to(device)
                    outputs = net(testImgs)
                    loss = lossF(outputs,labels)
                    predictions = torch.argmax(outputs,dim = 1)
                    #存储测试结果
                    totalLoss += loss
                    correct += torch.sum(predictions == labels)

                #计算总测试的平均准确率
                testAccuracy = correct/(BATCH_SIZE * len(testDataLoader))
                #计算总测试的平均Loss
                testLoss = totalLoss/len(testDataLoader)
                #将本step结果进行可视化处理
                processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %             
                                (epoch,EPOCHS,loss.item(),accuracy.item(),testLoss.item(),testAccuracy.item()))
    
    processBar.close()

#对测试Loss进行可视化
matplotlib.pyplot.plot(history['Test Loss'],label = 'Test Loss')
matplotlib.pyplot.legend(loc='best')
matplotlib.pyplot.grid(True)
matplotlib.pyplot.xlabel('Epoch')
matplotlib.pyplot.ylabel('Loss')
matplotlib.pyplot.show()

#对测试准确率进行可视化
matplotlib.pyplot.plot(history['Test Accuracy'],color = 'red',label = 'Test Accuracy')
matplotlib.pyplot.legend(loc='best')
matplotlib.pyplot.grid(True)
matplotlib.pyplot.xlabel('Epoch')
matplotlib.pyplot.ylabel('Accuracy')
matplotlib.pyplot.show()

torch.save(net,'./study/model.pth')
