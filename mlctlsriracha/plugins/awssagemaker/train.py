from sriracha.interfaces.TrainInterface import Train

class AwsSageMakerTrain(Train):

    def __init__(self, profile=None):
        print('Selected Azure ML profile')

    def getBucketNameFrom(gs_uri: str):
        bucket_start = -1
        for i in range(0, 2):
            bucket_start = gs_uri.find('/', bucket_start + 1)

        bucket_end = gs_uri.find('/', bucket_start + 1)
            
        # Printing nth occurrence
        # print ("Nth occurrence is at", val)

        bucket = gs_uri[bucket_start + 1: bucket_end]
        

        prefix = gs_uri[bucket_end + 1: len(gs_uri)]

        # chop off '/' at the end
        if prefix[len(prefix)-1: len(prefix)] == '/':
            prefix = prefix[0: len(prefix) - 1] 
        return bucket, prefix

    def inputAsDataframe(channel: str):
        """
        The function returns a dataframe for the input channel artifacts.

        GCP Vertex allows you to create a dataset that will be split into 
        train, test, and validation "channels" when loaded into your training run.
        This function handles merging the file structure that Vertex 
        exposes to the training container.

        Arguments:
            channel (str): The name of the channel which contains the given filename

        Returns:
            path (str): The absolute path to the specified channel file
        """

        data_directories = {
            'training': "AIP_TRAINING_DATA_URI",
            'validation': "AIP_VALIDATION_DATA_URI",
            'testing': "AIP_TEST_DATA_URI"
        }

        if channel in data_directories:
            print(f'Retrieving {channel} directory')
            gs_uri = os.getenv(data_directories[channel])
            storage_client = storage.Client()

            bucket, prefix=getBucketNameFrom(gs_uri)
            # chop off * for wildcard
            prefix = prefix.replace('*', '')
            print(f'bucket_name={bucket}')
            print(f'prefix={prefix}')
            # Note: Client.list_blobs requires at least package version 1.17.0.
            blobs = storage_client.list_blobs(bucket, prefix=prefix)
            # print(f'blobs: {blobs}')
            fileBytes = []
            for blob in blobs:
                print(f'blob.name: {blob.name}')
                # assumes CSV files
                s=str(blob.download_as_bytes(),'utf-8')
                data = StringIO(s) 
                df=pd.read_csv(data)
                fileBytes.append(df)

            frame = pd.concat(fileBytes, axis=0, ignore_index=True)     
            return frame
        else:
            print('Incorrect data channel type. Options are training, validation, and testing.')
            return null
    
    def log_artifact(filename):
        """
        The path to the output artifacts.

        Your algorithm should write all final model artifacts to this directory.
        Amazon SageMaker copies this data as a single object in compressed tar
        format to the S3 location that you specified in the CreateTrainingJob
        request. If multiple containers in a single training job write to this
        directory they should ensure no file/directory names clash. Amazon SageMaker
        aggregates the result in a tar file and uploads to S3.

        Arguments:
            filename (str): The name of the file which will be written back to S3

        Returns:
            path (str): The absolute path to the model output directory
        """
        return os.path.join(os.sep, 'opt', 'ml', 'model', filename)

    def finish():
        pass