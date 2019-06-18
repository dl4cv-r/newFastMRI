import json
import time


def create_submission_file(
  json_out_file, challenge, submission_url, model_name, model_description, nyu_data_only,
  participants=None, paper_url=None, code_url=None
):
    """
    Creates a JSON file for submitting to the leaderboard.
    You should first run your model on the test data, save the reconstructions, zip them up,
    and upload them to a cloud storage service (like Amazon S3).

    Args:
        json_out_file (str): Where to save the output submission file
        challenge (str): 'singlecoil' or 'multicoil' denoting the track
        submission_url (str): Publicly accessible URL to the submission files
        model_name (str): Name of your model
        model_description (str): A longer description of your model
        nyu_data_only (bool): True if you only used the NYU fastMRI data, False if you
            used external data
        participants (list[str], optional): Names of the participants
        paper_url (str, optional): Link to a publication describing the method
        code_url (str, optional): Link to the code for the model
    """

    if challenge not in {'singlecoil', 'multicoil'}:
        raise ValueError(f'Challenge should be singlecoil or multicoil, not {challenge}')

    phase_name = f'{challenge}_leaderboard'
    submission_data = dict(
        recon_zip_url=submission_url,
        model_name=model_name,
        model_description=model_description,
        nyudata_only=nyu_data_only,
        participants=participants,
        paper_url=paper_url,
        code_url=code_url
    )
    submission_data = dict(result=[{
        phase_name: submission_data
    }])

    with open(json_out_file, 'w') as json_file:
        json.dump(submission_data, json_file, indent=2)


if __name__ == '__main__':
    time_string = time.strftime('%Y-%m-%d_%H-%M', time.localtime(time.time()))
    submission_dir = f'./submission_{time_string}.json'
    # A url that allows direct downloads by machines. Use Google Cloud Platform or AWS, not Google Drive, etc.
    direct_download_url = 'https://storage.cloud.google.com/fastmri-challenge-submissions/my-submissions/adam_recon_val_normalized_ssim.zip'
    name = 'UNET'
    description = ''' Changed a few training tactics. '''
    members = ['veritas9872']

    create_submission_file(json_out_file=submission_dir, challenge='multicoil', submission_url=direct_download_url,
                           model_name=name, model_description=description, nyu_data_only=True, participants=members)
