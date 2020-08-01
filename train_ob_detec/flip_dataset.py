import xml.etree.ElementTree as ET
import glob
import cv2
import argparse

def flipxml (file_in, file_out):
    tree = ET.parse(file_in)
    root = tree.getroot()
    filename = file_out.split('/')[-1].split('.')[0] + '.jpg'
    root.find('filename').text = filename

    for elem in root.iter('polygon'):
        points = elem.findall('pt')
        for pt in points:
            elem.remove(pt)
            x = pt.find('x')
            x.text = str(150-int(x.text))

        elem.append(points[1])
        elem.append(points[0])
        elem.append(points[3])
        elem.append(points[2])

    tree.write(file_out)

def flipImg (file_in, file_out):
    img = cv2.imread(file_in)

    flipHorizontal = cv2.flip(img, 1)

    cv2.imwrite(file_out, flipHorizontal)

def flip(fileimg, ext, dir_in, dir_out):
    filexml = fileimg.split(ext)[0] + '.xml'
    filepath = fileimg.split(dir_in)[1]
    filename = filepath.split(ext)[0]
    filenumber = filename.split('person')[1]

    fileout = dir_out + 'person02' + filenumber
    fileoutxml = fileout+'.xml'
    fileoutimg = fileout + ext

    flipImg(fileimg, fileoutimg)
    flipxml(filexml,fileoutxml)


def main ():
    parser = argparse.ArgumentParser(description="flip images of dataset",
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '-e', '--ext',
        help='extension of images.',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-i', '--imageDir',
        help='Path to the folder where the image dataset is stored.',
        type=str,
        required=True
    )
    parser.add_argument(
        '-o', '--outputDir',
        help='Path to the output folder where the train and test dirs should be created. '
             'Defaults to the same directory as IMAGEDIR.',
        type=str,
        default=None
    )
    args = parser.parse_args()
    if args.outputDir is None:
        args.outputDir = args.imageDir
    ext = '.' + args.ext

    files = glob.glob(args.imageDir + '*' + ext)
    print(files)

    for f in files:
        flip(f, ext, args.imageDir, args.outputDir)


if __name__ == '__main__':
    main()


    
