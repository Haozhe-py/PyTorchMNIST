import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize
from PIL import Image
import numpy as np
import sys
import model as m

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

def read_png(fpath:str) -> np.array:
    try:
        if fpath[0] == '\"' and fpath[-1] == '\"':
            fpath = fpath[1:-1]
        elif fpath[0] == '\'' and fpath[-1] == '\'':
            fpath = fpath[1:-1]
        img = Image.open(fpath).resize((28,28)).convert('L')
        result = 255.0 - np.array(img, dtype = 'float32' )
        result = np.array([[result.tolist()]], dtype = 'float32')
        #print(result.shape)
        return result
    except:
        print('ERROR: Cannot load file',file=sys.stderr)

def run() -> None:
    model = torch.load('model.pt', weights_only = False)
    norm = Normalize(mean = [0.1307], std = [0.3081])
    print('欢迎使用TensorAI（PyTorch版本），此程序可识别图片中的手写数字。（仅限0~9）\n\n')
    while True:
        try:
            path = input('请输入图片文件（PNG格式）的路径。（输入exit退出）\n路径：')
            if path == 'exit':
                break
            else:
                input_p = torch.from_numpy(read_png(path))
                norm_p = norm(input_p)
                out = model(norm_p)
                out = F.softmax(out, dim=1)
                result = torch.argmax(out, dim=1).numpy().tolist()[0]
                print('识别结果：', result, '\n\n')
                continue
        except KeyboardInterrupt:
            print('ERROR: KeyboardInterrupt\n\n', file = sys.stderr)
        except Exception as e:
            print(f'ERROR: {str(e)}\n\n', file = sys.stderr)
    return None

if __name__ == '__main__':
    run()
    exit()