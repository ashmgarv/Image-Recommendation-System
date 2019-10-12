import numpy as np
from pymongo import MongoClient
import mahotas as mh
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape

class LBP:
    def __init__(self, images_folder):
        mongo_connection = MongoClient('mongodb://localhost:27017/')
        self.db = mongo_connection['mwdbphase1']
        self.images_path = images_folder

    def store_lbp(self):
        for file in os.listdir(self.images_path):
            lbp_array = []

            # Read the image as gray scale
            img1 = mh.imread(self.images_path + '/' + file, as_grey=True)

            # Turn the image into 100 * 100 smaller np arrays
            hch = LBP.turn_into_100c100(img1, 100, 100)

            # Perform lbp using 1 as the radius and 8 as the number of points to be considered for each sub_array
            # Returns a histogram of features for each sub_array
            for row in hch:
                lbp_array.append(mh.features.lbp(row, 1, 8))
            nps = np.concatenate(lbp_array, axis=0)

            # As a result, we get a single vector with 192 values as a feature descriptor
            # Now store output of each image a mongo collection
            self.db['lbp'].insert_one({'id': str(file.split('.')[0]), 'lbp_val': list(nps)})

    @staticmethod
    def turn_into_100c100(arr, nrows, ncols):
        h, w = arr.shape
        return (arr.reshape(h // nrows, nrows, -1, ncols).swapaxes(1, 2).reshape(-1, nrows, ncols))

    @staticmethod
    def cosine_similarity(a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        ans = dot_product / (norm_a * norm_b)
        return ans

    # Method to match and visualize results based on matches found
    def query_and_visualise(self, img_id, model, k):
        cosine_sims = []
        if model == "lbp":
            out_file = open(os.getcwd() + '/Output/lbp_search_results.txt', 'w+')

            # Get details of the provided image id
            cur_img = list(self.db['lbp'].find({'id': img_id}))

            # Query all other images
            all_data = list(self.db['lbp'].find())
            images_to_display = []

            # Reshape the 1d vec to a 2d one for skikit learn to work properly
            cur_img_lbp_val = np.array(cur_img[0]['lbp_val'])

            # Compute the cosine similarity for all images with respect to the given image
            for data in all_data:
                img = data['id']
                if img_id != img:
                    cosine_sims.append(
                        [img, LBP.cosine_similarity(cur_img_lbp_val, np.array(data['lbp_val']))])

            # Sort and get the top k images
            cosine_sims.sort(key=lambda a: a[1], reverse=True)
            to_plot = cosine_sims[:k]
            for image in to_plot:
                images_to_display.append(image[0] + '.jpg')

            # Print the values
            out_file.write("--------------------LBP SEARCH RESULTS--------------------\n")
            out_file.write(f"Overall Matching Scores for the {k} images for the image-id {img_id} are as follows\n")
            out_file.write('---------------------------------------------------\n')
            print(f"\n\nOverall Matching Scores for the {k} images for the image-id {img_id} are as follows")
            for index, val in enumerate(to_plot):
                out_file.write('Image No :' + str(index + 1) + "\n")
                print('Image No :' + str(index + 1))
                out_file.write('Image-Id : ' + str(val[0]) + '-- Matching Score : ' + str(val[1]) + "\n")
                print('Image-Id : ' + str(val[0]), '-- Matching Score : ' + str(val[1]))
                out_file.write('---------------------------------------------------\n')
                print('---------------------------------------------------')
            out_file.close()

            # Generate the html file
            LBP.load_template(img_id + '.jpg', images_to_display, self.images_path)

    # Generate the output html file
    @staticmethod
    def load_template(source_image, images_to_show, images_path):
        template_loader = FileSystemLoader(searchpath="./")
        env = Environment(
            loader=template_loader,
            autoescape=select_autoescape(['html', 'xml'])
        )
        template = env.get_template('template.html')
        optext = template.render({'source_image': source_image, 'images': images_to_show,
                                  'images_path': os.getcwd() + '/' + images_path})
        html_file = open('Output/k_similar_images.html', 'w+')
        html_file.write(optext)
        html_file.close()

