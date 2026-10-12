"""Inspectible two-convolution MIL draft; compilation/placement remain unverified."""


def projection(k, n, rank=None, allocated_s=32):
    if any(type(v) is not int or v <= 0 for v in (k, n, allocated_s)):
        raise ValueError('invalid shape')
    if rank is not None and (type(rank) is not int or not 1 <= rank <= min(k, n)):
        raise ValueError('invalid rank')
    def tensor(c):
        return f'tensor<fp16, [1, {c}, 1, {allocated_s}]>'
    def weight(name, out, inp):
        return (f'        tensor<fp16, [{out}, {inp}, 1, 1]> {name} = const()'
                f'[name = tensor<string, []>("{name}"), val = tensor<fp16, [{out}, {inp}, 1, 1]>'
                f'(BLOBFILE(path = tensor<string, []>("@model_path/weights/{name}.bin"), '
                f'offset = tensor<uint64, []>(64)))];')
    def conv(name, out, w, x):
        return (f'        {tensor(out)} {name} = conv(dilations = dl, groups = gr, pad = pd, '
                f'pad_type = pt, strides = st, weight = {w}, x = {x})'
                f'[name = tensor<string, []>("{name}")];')
    lines = ['program(1.0)', '{', f'    func main<ios16>({tensor(k)} x) {{',
             '        tensor<string, []> pt = const()[name = tensor<string, []>("pt"), val = tensor<string, []>("valid")];',
             '        tensor<int32, [2]> st = const()[name = tensor<string, []>("st"), val = tensor<int32, [2]>([1,1])];',
             '        tensor<int32, [4]> pd = const()[name = tensor<string, []>("pd"), val = tensor<int32, [4]>([0,0,0,0])];',
             '        tensor<int32, [2]> dl = const()[name = tensor<string, []>("dl"), val = tensor<int32, [2]>([1,1])];',
             '        tensor<int32, []> gr = const()[name = tensor<string, []>("gr"), val = tensor<int32, []>(1)];']
    if rank is None:
        lines += [weight('W', n, k), conv('y', n, 'W', 'x')]
    else:
        lines += [weight('A', rank, k), weight('B', n, rank),
                  conv('z', rank, 'A', 'x'), conv('y', n, 'B', 'z')]
    return '\n'.join(lines + ['    } -> (y);', '}']) + '\n'
