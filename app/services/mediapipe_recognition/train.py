import numpy as np
import torch as t
from .model import HandModel
from torch import nn
from torchnet import meter
from torch.autograd import Variable
import copy
from .tools.common import labels

label = labels
label_num = len(label)
# label saving address: label+.npz
targetX = [0 for xx in range(label_num)]
target = []
for xx in range(label_num):
    target_this = copy.deepcopy(targetX)
    target_this[xx] = 1
    target.append(target_this)

lr = 1e-3  # learning rate
model_saved = 'checkpoints/model'

model = HandModel(num_classes=label_num)
optimizer = t.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

loss_meter = meter.AverageValueMeter()

epochs = 40
for epoch in range(epochs):
    print("epoch:" + str(epoch))
    loss_meter.reset()
    count = 0
    allnum = 0
    for i in range(len(label)):
        data = np.load('./npz_files/' + label[i] + ".npz", allow_pickle=True)
        data = data['data']

        for j in range(len(data)):
            xdata = t.tensor(data[j]).unsqueeze(0)
            optimizer.zero_grad()
            this_target = t.tensor([target[i].index(1)])
            input_, this_target = Variable(xdata), Variable(this_target)

            output = model(input_)

            output_index = output.argmax().item()
            outLabel = label[output_index]
            targetIndex = target[i].index(1)
            targetLabel = label[targetIndex]
            if targetLabel == outLabel:
                count += 1
            allnum += 1

            loss = criterion(output, this_target)
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.data)

    print("correct_rate:", str(count / allnum))

    t.save(model.state_dict(), '%s_%s.pth' % (model_saved, epoch))