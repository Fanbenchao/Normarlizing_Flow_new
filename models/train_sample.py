import time
import queue
from utils.sidd_utils import sample_sidd_tf,calc_kldiv_mb
import numpy as np
import os
def take_batch(batch_list):
    _x = batch_list[0]['in']
    _y = batch_list[0]['gt']
    _nlf0 = batch_list[0]['nlf0']
    _nlf1 = batch_list[0]['nlf1']
    _iso = batch_list[0]['iso']
    _shutter = batch_list[0]['shutter']
    for i in range(1,len(batch_list)):
        _x = np.concatenate([_x,batch_list[i]['in']],axis = 0)
        _y = np.concatenate([_y,batch_list[i]['gt']],axis = 0)
        _nlf0 = np.concatenate([_nlf0,batch_list[i]['nlf0']],axis = 0)
        _nlf1 = np.concatenate([_nlf1,batch_list[i]['nlf1']],axis = 0)
        _iso = np.concatenate([_iso,batch_list[i]['iso']],axis = 0)
        _shutter = np.concatenate([_shutter,batch_list[i]['shutter']],axis = 0)
    return _x,_y,_nlf0,_nlf1,_iso,_shutter

def sample(sess,nf,test_list,x,y,iso,shutter,is_training,batch_size,step,hps,sample_logger,epoch):
    t = time.time()
    kldiv3 = np.zeros(4)
    is_cond = hps.sidd_cond != 'uncond'
    x_sample = sample_sidd_tf(sess, nf, is_training, hps.temp, y, iso,is_cond, shutter)
    for k in range(step):
        if k == step-1:
            test_batch = test_list[k*batch_size:]
        else:
            test_batch = test_list[k*batch_size:(k+1)*batch_size]
        
        _,_y,_,_,_iso,_shutter = take_batch(test_batch)
              # sample (forward)
        x_sample_val = sess.run(x_sample, feed_dict={y: _y,iso: _iso,shutter:_shutter, is_training: False})
        # marginal KL divergence
        vis_mbs_dir = os.path.join(hps.logdir, 'samples_epoch_%04d' % epoch, 'samples_%.1f' % hps.temp)
        kl = calc_kldiv_mb(test_batch, x_sample_val, vis_mbs_dir)
        kldiv3 +=kl
    kldiv3 /= step
    t_sample = time.time() - t

    # log
    log_dict = {'epoch': epoch, 'sample_time': t_sample, 'KLD_G': kldiv3[0],
                'KLD_NLF': kldiv3[1], 'KLD_NF': kldiv3[2], 'KLD_R': kldiv3[3]}
    sample_logger.log(log_dict)
    return kldiv3,t_sample

def train(sess,train_list,loss_val,sd_z,train_op,x,y,nlf0,nlf1,iso,shutter,lr,is_training,
          batch_size,step,hps,train_logger,epoch):
    t = time.time()
    train_epoch_loss = 0
    sd_z_tr = 0
    
    for k in range(step):
        if k == step-1:
            train_batch = train_list[k*batch_size:]
        else:
            train_batch = train_list[k*batch_size:(k+1)*batch_size]
        
        _x,_y,_nlf0,_nlf1,_iso,_shutter= take_batch(train_batch)
        if hps.sidd_cond == 'condSDN':
            train_loss, sd_z_val = sess.run(
                [loss_val, sd_z], feed_dict={x: _x, y: _y, nlf0: _nlf0, nlf1: _nlf1, iso: _iso,shutter:_shutter,
                                         lr: hps.lr, is_training: True})
        else:
            _, train_loss, sd_z_val = sess.run(
                [train_op, loss_val, sd_z], feed_dict={x: _x, y: _y, nlf0: _nlf0, nlf1: _nlf1, iso: _iso,shutter:_shutter,
                                                    lr: hps.lr,is_training: True})
        sd_z_tr += sd_z_val 
        train_epoch_loss += train_loss
    sd_z_tr /= step
    train_epoch_loss /= step
    t_train = time.time() - t

    train_logger.log({'epoch': epoch, 'train_time': int(t_train),
                      'NLL': train_epoch_loss,'sdz': sd_z_tr})
    return sd_z_tr,train_epoch_loss,t_train



        
        