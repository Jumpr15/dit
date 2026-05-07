import os
import shutil

#Path of source directory & destination directory
src_directory = '/danbooru_faces/danbooru_eva_faces'
dst_directory = 'images'

# Extract file from Source directory and copy to Destination directory
for file_num in range(4):   
     src_directory = f'danbooru_faces/danbooru_eva_faces/{file_num}'
     for file in os.listdir(src_directory):
          src_file = os.path.join(src_directory, file)
          dest_file = os.path.join(dst_directory, file)

          shutil.copy(src_file, dest_file)