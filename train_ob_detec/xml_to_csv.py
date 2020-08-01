import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import argparse

# for object-detection-api
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            x = [int(i.text) for i in member.findall('.//x')]
            y = [int(i.text) for i in member.findall('.//y')]
            value = (root.find('filename').text,
                     int(root.find('imagesize')[1].text),
                     int(root.find('imagesize')[0].text),
                     member[0].text,
                     min(x),
                     min(y),
                     max(x),
                     max(y)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

# for keras-retinanet
def xml_to_csv_v2(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            x = [int(i.text) for i in member.findall('.//x')]
            y = [int(i.text) for i in member.findall('.//y')]
            value = ('images/' + root.find('filename').text,
                     min(x),
                     min(y),
                     max(x),
                     max(y),
                     member[0].text
                     )
            xml_list.append(value)
    column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="TensorFlow XML(LABELME)-to-CSV converter")
    parser.add_argument("-i",
                        "--inputDir",
                        help="Path to the folder where the input .xml files are stored",
                        required=True,
                        type=str)
    parser.add_argument("-o",
                        "--outputFile",
                        help="Name of output .csv file (including path)", type=str)
    args = parser.parse_args()

    if(args.outputFile is None):
        args.outputFile = args.inputDir + "/labels.csv"

    assert(os.path.isdir(args.inputDir))

    xml_df = xml_to_csv(args.inputDir)
    xml_df.to_csv(
        args.outputFile, index=None)
    # for keras-retinanet
    # xml_df.to_csv(
    #     args.outputFile, index=None, header=None)
    print('Successfully converted xml to csv.')

if __name__ == '__main__':
    main()
