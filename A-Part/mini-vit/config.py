"""
    bs(int): batchsize, batchsize of data
    mt(float): momentum, momentum of optimizier
    dm(int): dim, after embedding dim
    dp(int): depth, depth of multi head
    hs(int): heads, head num
    md(int): mlp_dim, feed forward hidden layer dim
"""
config_list =  ['bs',  'mt',  'dm',  'dp',  'hs',  'md',]
config_para = [
                [  96,   0.9,     128,    2,     4,     256,],
                [  32,   0.9,     768,    2,     6,    1024,],
                [  48,   0.9,    1152,    2,     5,    1792,],
                [ 256,   0.9,      32,    2,     4,      64,],
                [1024,   0.9,      64,    2,     2,      64,],
              ]
configs = []
for i in range(len(config_para)):
    dict_temp = {'batch_size':config_para[i][0],
                 'momentum':config_para[i][1],
                 'dim':config_para[i][2],
                 'depth':config_para[i][3],
                 'heads':config_para[i][4],
                 'mlp_dim':config_para[i][5],
    }
    configs.append(dict_temp)


