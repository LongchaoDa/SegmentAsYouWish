# Instructions: 

Please run the demo task:

1. use the script `predictor.py` to generated predicted results.
2. in the this script, please modify below path: 
    parser.add_argument('--data_root_path', default="data_temp/simulate/npy/CT_Abd", help='data root path') 
    parser.add_argument('--result_save_path', default="data_temp/simulate/npy/prediction", help='path for save result')

3. modify the organ_list_to_seg = [2, 3, 6], by checking the Name_DIC with correct list. 
4. once you executed the code, then it will output the generated predictions in the correspondings foler, the subfolder name is the same as the input file name, and then within the subfolder, there will be n .nii.gz files, in the same amount as the length of the organ_list_to_seg.

5. then you can run teh `predictor_analysis.ipynb`, it has visulization on the raw image, groundtruth, and the predicted result. and you can choose which slice you wanna see. Also, you can calculate the dice_scores_mean (see the example in code. please feel free to modify it)


