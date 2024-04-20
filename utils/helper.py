import os
import shutil

race_folder_path = '/Users/viru/Documents/GitHub/MixFairFace-Image-Recognization/resources/newtrainingdata/test'

races_list = os.listdir(race_folder_path)

identities = 1000
classes = {'Caucasian': 0, 'Indian': 1, 'Asian': 2, 'African': 3}

for race in races_list:

    if race != '.DS_Store':
        curr_race_path = os.path.join(race_folder_path, race)
        curr_class = classes[race]

        identity_list = os.listdir(curr_race_path)

        for idt in identity_list:
            if idt != '.DS_Store':
                curr_identity = os.path.join(curr_race_path, idt)
                images = os.listdir(curr_identity)
                print(len(images), curr_identity)
                
                train_data_path = '/Users/viru/Documents/GitHub/MixFairFace-Image-Recognization/resources/data/test'

                train_class_path = os.path.join(train_data_path, str(curr_class))

                # Copy train images
                for img in images:
                    src = os.path.join(curr_identity, img)
                    dst = os.path.join(train_class_path, img)
                    shutil.copy(src, dst)
