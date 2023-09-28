import tarfile

tar_file_path = 'data/DataSets.tar'
extraction_dir = 'data/extracted_data/'

try:
    with tarfile.open(tar_file_path, 'r') as tar:
        tar.extractall(path=extraction_dir)
    extraction_success = True
except Exception as e:
    extraction_success = False
    error_message = str(e)

extraction_success, error_message if not extraction_success else "Done"
