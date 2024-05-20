from PIL import Image
import numpy as np
import hashlib
import torch



def custom_hash(passport_img=None, hash_length=512):
    if hash_length > 512:
        return "Hash length must be less than or equal to 512 bits"
    if passport_img is None:
        image_path='./data/trigger_set/pics/0.jpg'
        # Open the image
        with open(image_path, 'rb') as f:
            # Read the image
            image_data = f.read()
    else:
        image_data = passport_img.cpu().numpy().tobytes()
    
    # Generate a hash of the image using SHA-512 (512 bits)
    m = hashlib.sha512()
    m.update(image_data)
    hex_hash = m.hexdigest()
    
    # Convert hex hash to binary string
    full_binary_hash = bin(int(hex_hash, 16))[2:].zfill(512)  # SHA-512 hash is 512 bits
    
    # Truncate or pad the binary string to the specified length
    if len(full_binary_hash) >= hash_length:
        truncated_hash = full_binary_hash[:hash_length]
    else:
        print("hash overlength", len(full_binary_hash))
        num_repeats = (hash_length + len(full_binary_hash) - 1) // len(full_binary_hash)
        full_binary_hash = full_binary_hash * num_repeats
        truncated_hash[:] = full_binary_hash[:hash_length]

        #truncated_hash = full_binary_hash.ljust(hash_length, '0')
    
    binary_hash = list(map(int, truncated_hash))
    binary_hash = torch.tensor(binary_hash)

    binary_hash = torch.sign(binary_hash - 0.5)
    try:
        assert len(binary_hash) == hash_length
    except:
        print('Invalid binary hash length for the passport signature!, see models/layers/hash.py')
        exit()
    
    return binary_hash


if __name__ == '__main__':
    # Load an image
    import torchvision.transforms as transforms 

    image_path_3 = './data/trigger_set/pics/0.jpg'
    image_path_4 = './data/trigger_set/pics/1.jpg'
    image3 = Image.open(image_path_3)
    image4 = Image.open(image_path_4)

    # Generate the hash code

    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    transform = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #transform = transforms.Compose([transforms.ToTensor()])
    image3 = transform(image3).unsqueeze(0)
    image4 = transform(image4).unsqueeze(0)

    bit_num = 0
    cs = []
    hash_3 = custom_hash(image3, 512)
    hash_4 = custom_hash(image4, 512)

    for d1, d2 in zip(hash_3, hash_4):
        # d1 = d1.view(d1.size(0), -1)
        # d2 = d2.view(d2.size(0), -1)
        cs.append((d1 != d2).sum().item())
        bit_num += 1
    print(f'Bit error rate of Real and Fake signature: {sum(cs)  / bit_num:.4f}')
