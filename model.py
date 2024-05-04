import torch
from torch import nn


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x

# A simple debug test, you can always do a simple debug test to make sure your model is working
# 一个简单的debug test, 你可以总是通过是实例化一个模型，然后输入一个随机的tensor来测试你的模型是否正常工作
# model = MyAwesomeModel()
# dummy_input = torch.randn(1, 1, 28, 28) # batch_size, channels, height, width， 1个样本， 1个通道， 28*28的图片
# 这个dummy input和真实图片的shape是一样的， 但是里面的值是随机的， 而真实的图片是有意义的
# output = model(dummy_input)
# print(output.shape, "This is the output shape of the image after passing through the model")

if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    torch.save(model.state_dict(), "model.pt")  
    # 将PyTorch模型的参数保存到文件中。具体来说，model.state_dict()返回一个字典，其中包含了模型所有的参数。
    # 这些参数可以是神经网络中的权重、偏置等。然后，torch.save()函数将这个字典保存到名为"model.pt"的文件中。
    # 允许你在之后重新加载模型的参数，而无需重新训练模型。通常情况下，你可以使用torch.load()函数加载这个保存
    # 的参数字典，并将它们加载到相应的模型中。
 
    # # 这个模型的最终输出是[1, 10],1 代笔了一个 batch size 为1的样本， 10代表了10个类别的概率分布，
    # 这里是10是因为我们的数据集是MNIST， 有10个类别， 分别是0-9
    # 因为我们使用的是softmax， 所以这个输出是概率分布， 一个例子可能是[0.1, 0.8, 0.05, 0.05, 0, 0, 0, 0, 0, 0]
    # 你可以通过argmax来获取最大的概率值， 也就是预测的类别
    # 在我们这个简单的例子中， argmax是1， 因为0.8是最大的预测概率， 也就是第二个类别， 也就是数字1

    