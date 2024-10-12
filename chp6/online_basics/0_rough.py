
def postprocess_image():
    from torchvision import transforms
    postprocess = transforms.ToPILImage()
    print(type(postprocess))
postprocess_image()