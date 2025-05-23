from data_provider.data_loader import Dataset_PEMS_graph
from torch.utils.data import DataLoader

data_dict = {
    'PEMS_graph': Dataset_PEMS_graph,
}

def data_provider(args, flag):
    Data = data_dict[args.data]

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    print(f'data_provider data_path: {args.data_path}')
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        args=args,
        scale=False,
        )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,)
    return data_set, data_loader
