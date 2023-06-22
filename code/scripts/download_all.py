import subprocess
import sys
import os
import concurrent.futures


# List of files to download
csv_files = [
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Ar_ITA_train.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Ar_ITA_test.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Ar_ITA_val.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Bu_US-PNA_train.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Bu_US-PNA_test.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Bu_US-PNA_val.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Cl_US-MN_train.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Cl_US-MN_test.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Cl_US-MN_val.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Ga_US-PNA_train.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Ga_US-PNA_test.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Ga_US-PNA_val.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Kn_IND_train.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Kn_IND_test.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Kn_IND_val.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Sh_IND_train.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Sh_IND_test.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Sh_IND_val.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Pl_US-IN_train.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Pl_US-IN_test.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Pl_US-IN_val.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Sp_COL_train.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Sp_COL_test.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Sp_COL_val.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Me_SAU_train.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Me_SAU_test.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Me_SAU_val.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Co_JPN_train.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Co_JPN_test.csv",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_csv/Co_JPN_val.csv"
]

ffcv_files = [
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Arts_and_craft_ITA_train.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Arts_and_craft_ITA_test.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Arts_and_craft_ITA_val.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Building_US-PNA_train.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Building_US-PNA_test.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Building_US-PNA_val.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Cleaning_US-MN_train.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Cleaning_US-MN_test.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Cleaning_US-MN_val.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Gardening_US-PNA_train.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Gardening_US-PNA_test.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Gardening_US-PNA_val.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Knitting_IND_train.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Knitting_IND_test.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Knitting_IND_val.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Shopping_IND_train.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Shopping_IND_test.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Shopping_IND_val.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Playing_US-IN_train.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Playing_US-IN_test.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Playing_US-IN_val.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Sport_COL_train.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Sport_COL_test.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Sport_COL_val.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Mechanic_SAU_train.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Mechanic_SAU_test.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Mechanic_SAU_val.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Cooking_JPN_train.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Cooking_JPN_test.beton",
    "s3://ego4d-bristol/public/ARGO1M/ARGO1M_ffcv/Cooking_JPN_val.beton"
]

# Determine the flag and directory provided by the user
if len(sys.argv) < 3:
    print("Please specify a flag (--csv or --ffcv) and a directory to download the files.")
    sys.exit(1)

flag = sys.argv[1]
directory = sys.argv[2]

# Validate the flag
if flag == "--csv":
    files = csv_files
elif flag == "--ffcv":
    files = ffcv_files
else:
    print("Invalid flag. Please specify either --csv or --ffcv")
    sys.exit(1)

# Get the absolute path of the directory
directory = os.path.abspath(directory)

# Create the directory if it doesn't exist
os.makedirs(directory, exist_ok=True)

def download_file(file):
    # Execute the AWS CLI command to download the file to the specified directory
    command = f"aws s3 cp {file} {directory}"
    subprocess.run(command, shell=True, check=True)
    print(f"Downloaded {file} to {directory}")

# Set the maximum number of concurrent downloads (adjust this based on available resources)
max_workers = 5

# Create a ThreadPoolExecutor with the specified maximum number of workers
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit the download tasks to the executor
    futures = [executor.submit(download_file, file) for file in files]

    # Wait for all the download tasks to complete
    concurrent.futures.wait(futures)

print("All files downloaded successfully.")