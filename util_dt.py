# coding: utf-8
import numpy as np
import pandas as pd
import time
import datetime

# convert : object > array( float )
def conv_obj_dtArr(arr ):
    dim = pd.to_datetime(arr )
    dtList=[]
    #print(rdDim['time'][:10 ])
    #quit()
    for item in dim:
#        print(item )
        utm =time.mktime( item.timetuple())
        dtList.append(utm )
        #ux= time.mktime( dt_1.timetuple())
    #print(dtList )
    ret  =np.array(dtList).reshape(len(dtList),1)
    # dt= 2100
    dt_2100  = datetime.datetime(2100, 1, 1, 0, 0, 0)
    ux_2100 = time.mktime( dt_2100.timetuple())
    ret = ret / ux_2100
    return ret

#
# add arr, hh = add hours
# return np.array
def add_date_arr(arr, hh):
    dim = pd.to_datetime(arr )
    arrList=[]
    for item in dim:
        utm =time.mktime( item.timetuple())
#        loc = datetime.datetime.fromtimestamp(utm )
        arrList.append(utm )

    max_obj = arr.max()
    #print(arr.max())
    max_dt = pd.to_datetime(max_obj )
    utm =time.mktime( max_dt.timetuple())
    print(utm )
    next =utm + (hh * 3600 )
    loc = datetime.datetime.fromtimestamp(next )
#    loc = loc +hh
    #print(loc)
    span= next - utm
    span_mm = span / 60
    #print(span )
    add_arr = np.arange(utm, next , 60 *10 )
    #print(len(add_arr ))
#    dtList=[]
    for item in add_arr:
#        loc = datetime.datetime.fromtimestamp(item)
        arrList.append(item)
    arr_dt   =np.array(arrList ).reshape(len(arrList),1)
    # dt= 2100
    dt_2100  = datetime.datetime(2100, 1, 1, 0, 0, 0)
    ux_2100 = time.mktime( dt_2100.timetuple())
    arr_dt = arr_dt / ux_2100    
#    print(arr_dt[ : 100 ])
#    print(arr_dt.shape )
    return arr_dt

#
def conv_num_date(arr ):
    dtList=[]
    # dt= 2100
    dt_2100  = datetime.datetime(2100, 1, 1, 0, 0, 0)
    ux_2100 = time.mktime( dt_2100.timetuple())
#    x_train = x_train / ux_2100
#    print( x_train.shape )
    #print( x_train[: 10] )
    arr = arr * ux_2100  #ux_time
    for item in arr:
        loc = datetime.datetime.fromtimestamp(item)
        dtList.append(loc )
#        print(loc)
#    
    ret  =np.array(dtList).reshape(len(dtList),1)
    return ret

#
# return : x_train, x_test
def load_data(x_train   , ritu):
    batch_size   =int( len(x_train)*  ritu ) 
    batch_size_t =int( len(x_train)*  (1 - ritu) ) 
#    t_ritu = 1 -ritu
    batch_mask   = np.random.choice(len(x_train), batch_size )
    batch_mask_t = np.random.choice(len(x_train), batch_size_t )

    x_batch = x_train[batch_mask]    #x_train
    t_batch = x_train[batch_mask_t]  #
    return (x_batch , t_batch)
#
# return : x_train, x_test
def load_data_old(x_train   , ritu):
    batch_size =int( len(x_train)*  ritu ) 
    batch_mask = np.random.choice(len(x_train), batch_size )
    x_batch = x_train[batch_mask]  #x_train
    print(x_batch.shape )
    print( batch_mask )
    # test
    list=[]
    for i in range(len(x_train)):
    #    print(i)
        item =x_train[i]
        flg=0
        for j in batch_mask:
            if(i==j):
                flg=1
        if(flg==0):
            list.append(item)
    t_batch = np.array(list).reshape(len(list),1)
#    print(t_batch  )
    print(t_batch.shape )
    return (x_batch , t_batch)


def save_params(file_name="params.pkl"):
    params = {}
    for key, val in self.params.items():
        params[key] = val
    with open(file_name, 'wb') as f:
        pickle.dump(params, f)


