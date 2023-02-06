import torch
import os
from exp.exp_main import Exp_Main

class Params():
    def __init__(self, description='Transformer family for Time Series Forecasting', is_training=1, task_id='test',
                 model='TDformer', version='Fourier', mode_select='random', modes=64, L=3, base='legendre', cross_activation='tanh',
                 data='custom', root_path='data', data_path='BTC-USD(2019_8_1~2020_8_1).csv', features='S', target='Close', freq='d', checkpoints='checkpoints',
                 seq_len=20, label_len=10, pred_len=1, enc_in=1, dec_in=1, c_out=1, d_model=512, n_heads=4, e_layers=2, d_layers=1,
                 d_ff=2048, moving_avg=[24], factor=1, distil=True, dropout=0.05, embed='timeF', activation='softmax', output_attention=False,
                 do_predict=False, num_workers=0, itr=3, train_epochs=20, batch_size=32, patience=3, learning_rate=0.0001, des='test',
                 loss='mse', lradj='type1', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0', K=0, temp=1, adjust=False,
                 output_stl = False):
        self.description = description
        self.is_training = is_training
        self.task_id = task_id
        self.model = model
        self.version = version
        self.mode_select = mode_select
        self.modes = modes
        self.L = L
        self.base = base
        self.cross_activation = cross_activation
        self.data = data
        self.root_path = root_path
        self.data_path = data_path
        self.features=features
        self.target = target
        self.freq = freq
        self.checkpoints = checkpoints
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.moving_avg = moving_avg
        self.factor = factor
        self.distil = distil
        self.dropout = dropout
        self.embed = embed
        self.activation = activation
        self.output_attention = output_attention
        self.do_predict = do_predict
        self.num_workers = num_workers
        self.itr = itr
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.learning_rate = learning_rate
        self.des = des
        self.loss = loss
        self.lradj = lradj
        self.use_amp = use_amp
        self.use_gpu = use_gpu
        self.gpu = gpu
        self.use_multi_gpu = use_multi_gpu
        self.devices = devices
        self.K = K
        self.temp = temp
        self.adjust = adjust
        self.output_stl = output_stl


args = Params()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_modes{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_id,
            args.model,
            args.mode_select,
            args.modes,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        # if args.do_predict:
        #     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        #     exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                  args.model,
                                                                                                  args.data,
                                                                                                  args.features,
                                                                                                  args.seq_len,
                                                                                                  args.label_len,
                                                                                                  args.pred_len,
                                                                                                  args.d_model,
                                                                                                  args.n_heads,
                                                                                                  args.e_layers,
                                                                                                  args.d_layers,
                                                                                                  args.d_ff,
                                                                                                  args.factor,
                                                                                                  args.embed,
                                                                                                  args.distil,
                                                                                                  args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
