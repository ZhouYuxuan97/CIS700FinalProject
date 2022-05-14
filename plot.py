import matplotlib.pyplot as plt
import numpy as np
import torch


hyper_para = {
    'epochs': 100,
    'M': .5, # short
    'N': .5, # long
    'max_len': 512,
    'dropout': 0.75,
    'batch_size': 100,
    'use_tokens': False,
    'verbose': 1,
    'lr': 0.00005,
    'test samples': 1000
}

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(14, 10))

for i in range(3):
    t_loss_short =  torch.load('./loss/t_loss_short_rep{}.pt'.format(i))
    v_loss_short =  torch.load('./loss/v_loss_short_rep{}.pt'.format(i))
    t_loss_long =  torch.load('./loss/t_loss_long_rep{}.pt'.format(i))
    v_loss_long =  torch.load('./loss/v_loss_long_rep{}.pt'.format(i))
    g_norm_short = torch.load('./loss/g_norm_short_rep{}.pt'.format(i))
    g_norm_long = torch.load('./loss/g_norm_long_rep{}.pt'.format(i))
    test_loss = torch.load('./loss/test_loss_rep{}.pt'.format(i))

    # short, training
    plt.subplot(2, 4, 1)
    t_loss_short= [x for x in t_loss_short]
    # plt.scatter(np.arange(hyper_para['epochs']) + 1, [l.detach().numpy() for l in t_loss_short])
    plt.plot(np.arange(hyper_para['epochs']) + 1, [l for l in t_loss_short])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('training (short)')

    # short, validation
    plt.subplot(2, 4, 2)
    v_loss_short= [x for x in v_loss_short]
    # plt.scatter(np.arange(hyper_para['epochs']) + 1, [l.detach().numpy() for l in v_loss_short])
    plt.plot(np.arange(hyper_para['epochs']) + 1, [l for l in v_loss_short])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('validation (short)')

    # long, training
    plt.subplot(2, 4, 3)
    t_loss_long= [x for x in t_loss_long]
    # plt.scatter(np.arange(hyper_para['epochs']) + 1, [l.detach().numpy() for l in t_loss_long])
    plt.plot(np.arange(hyper_para['epochs']) + 1, [l for l in t_loss_long])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('training (long)')

    # long, validation
    plt.subplot(2, 4, 4)
    v_loss_long= [x for x in v_loss_long]
    # plt.scatter(np.arange(hyper_para['epochs']) + 1, [l.detach().numpy() for l in v_loss_long])
    plt.plot(np.arange(hyper_para['epochs']) + 1, [l for l in v_loss_long])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('validation (long)')

    # short, gradient
    plt.subplot(2, 4, 5)
    g_norm_short= [x for x in g_norm_short]
    # plt.scatter(np.arange(hyper_para['epochs']) + 1, [l.detach().numpy() for l in t_loss_short])
    plt.plot(np.arange(hyper_para['epochs']) + 1, [l for l in g_norm_short])
    plt.xlabel('epoch')
    plt.ylabel('gradient norm')
    plt.title('gradient (short)')

    # long, gradient
    plt.subplot(2, 4, 6)
    g_norm_long= [x for x in g_norm_long]
    # plt.scatter(np.arange(hyper_para['epochs']) + 1, [l.detach().numpy() for l in t_loss_short])
    plt.plot(np.arange(hyper_para['epochs']) + 1, [l for l in g_norm_long],label="rep_%d"%i)
    plt.xlabel('epoch')
    plt.ylabel('gradient norm')
    plt.title('gradient (long)')

    # test
    plt.legend()
    plt.subplot(2, 4, 7)
    # plt.scatter(np.arange(len(test_loss)) + 1, [l.detach().numpy() for l in test_loss])

    plt.scatter(np.arange(len(test_loss)) + 1, [l.item() for l in test_loss])
    plt.xlabel('the number of validation samples')
    plt.ylabel('loss')
    plt.title('test')

plt.tight_layout()
plt.show()
# plt.savefig('./plots/results.png')
plt.close()