import logging
import os
import time,requests

import boto3
import yt_dlp
from botocore.exceptions import ClientError

logger = logging.getLogger("video-indexer")


class VideoIndexerService:
    def __init__(self):
        self.bucket_name = os.getenv("AWS_BUCKET_NAME")
        self.region = os.getenv("AWS_REGION", "us-east-1")

        if not self.bucket_name:
            raise ValueError("AWS_BUCKET_NAME environment variable is required")

        self.s3_client = boto3.client('s3', region_name=self.region)
        self.transcribe_client = boto3.client('transcribe', region_name=self.region)
        self.rekognition_client = boto3.client('rekognition', region_name=self.region)

    def download_youtube_video(self, url, output_path="temp_video.mp4"):
        """Download YouTube video using yt_dlp."""
        logger.info(f"Downloading YouTube video: {url}")

        ydl_opts = {
            'format': 'best[ext==mp4]',
            'outtmpl': output_path,
            'quiet': True,
            'overwrites': True
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            logger.info("Download completed")
            return output_path
        except Exception as e:
            raise Exception(f"YouTube download failed: {str(e)}")



    def _ensure_bucket_exists(self):

        """
        Create S3 bucket if it doesn't exist.
        """
        try:
            if self.region == "us-east-1":
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            logger.info(f"Bucket '{self.bucket_name}' created.")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code in ('BucketAlreadyOwnedByYou', 'BucketAlreadyExists'):
                logger.debug("Bucket already exists, skipping creation.")
            else:
                raise


    def upload_video(self, video_path, video_name):

        """
        Upload video to S3 and return the S3 URI.
        """
        self._ensure_bucket_exists()

        s3_key = f"videos/{video_name}.mp4"
        self.s3_client.upload_file(video_path, self.bucket_name, s3_key)

        s3_uri = f"s3://{self.bucket_name}/{s3_key}"
        logger.info(f"Video uploaded to {s3_uri}")
        return s3_uri
    



    def wait_for_transcription(self, job_name, poll_interval=10, timeout=600):
        """
        Poll Transcribe job until complete. Returns transcript text.
        """
        logger.info(f"Waiting for transcription job: {job_name}")
        elapsed = 0

        while elapsed < timeout:
            response = self.transcribe_client.get_transcription_job(
                TranscriptionJobName=job_name
            )
            status = response['TranscriptionJob']['TranscriptionJobStatus']
            logger.info(f"Transcription status: {status}")

            if status == 'COMPLETED':
                transcript_url = response['TranscriptionJob']['Transcript']['TranscriptFileUri']
                text = requests.get(transcript_url).json()
                return text['results']['transcripts'][0]['transcript']

            if status == 'FAILED':
                reason = response['TranscriptionJob'].get('FailureReason', 'Unknown')
                raise Exception(f"Transcription failed: {reason}")

            time.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"Transcription job '{job_name}' timed out after {timeout}s")

    def wait_for_rekognition(self, job_id, job_type="label", poll_interval=10, timeout=600):
        """Poll Rekognition job until complete. job_type: 'label' or 'text'."""
        logger.info(f"Waiting for Rekognition {job_type} job: {job_id}")
        elapsed = 0

        while elapsed < timeout:
            if job_type == "label":
                response = self.rekognition_client.get_label_detection(JobId=job_id)
            elif job_type == "text":
                response = self.rekognition_client.get_text_detection(JobId=job_id)
            else:
                raise ValueError(f"Unknown job_type: {job_type}")

            status = response['JobStatus']
            logger.info(f"Rekognition {job_type} status: {status}")

            if status == 'SUCCEEDED':
                return response
            if status == 'FAILED':
                raise Exception(f"Rekognition {job_type} job failed: {job_id}")

            time.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"Rekognition job '{job_id}' timed out after {timeout}s")
    
    def start_transcription(self, video_name):
        """Start a Transcribe job for the uploaded video."""
        s3_uri = f"s3://{self.bucket_name}/videos/{video_name}.mp4"
        job_name = f"transcribe-{video_name}-{int(time.time())}"

        self.transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': s3_uri},
            MediaFormat='mp4',
            LanguageCode='en-US',
        )
        logger.info(f"Transcription job started: {job_name}")
        return job_name


    def start_rekognition(self, video_name):
        """Start Rekognition label + text detection jobs."""
        s3_obj = {'S3Object': {'Bucket': self.bucket_name, 'Name': f'videos/{video_name}.mp4'}}

        label_job = self.rekognition_client.start_label_detection(
            Video=s3_obj, MinConfidence=70
        )['JobId']

        text_job = self.rekognition_client.start_text_detection(
            Video=s3_obj
        )['JobId']

        logger.info(f"Rekognition jobs started — labels: {label_job}, text: {text_job}")
        return label_job, text_job


    def extract_data(self, transcript, label_response, text_response):
        """Combine transcript, labels and OCR into clean state dict."""
        labels = list(set([
            l['Label']['Name'] for l in label_response.get('Labels', [])
        ]))

        ocr_texts = [
            t['TextDetection']['DetectedText']
            for t in text_response.get('TextDetections', [])
            if t['TextDetection']['Type'] == 'LINE'
        ]

        return {
            "transcript": transcript,
            "ocr_text": ocr_texts,
            "video_metadata": {"detected_labels": labels},
            "error": []
        }


    def wait_for_processing(self, video_name, retries=5, delay=5):
        """Verify the video was successfully uploaded to S3."""
        s3_key = f"videos/{video_name}.mp4"
        logger.info(f"Verifying upload for: s3://{self.bucket_name}/{s3_key}")

        for attempt in range(1, retries + 1):
            try:
                response = self.s3_client.head_object(
                    Bucket=self.bucket_name, Key=s3_key
                )
                size = response['ContentLength']
                last_modified = response['LastModified']
                logger.info(f"✅ Upload verified — Size: {size/(1024*1024):.2f} MB, Last Modified: {last_modified}")
                return True

            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == '404':
                    logger.warning(f"Attempt {attempt}/{retries}: Not found yet, retrying in {delay}s...")
                    time.sleep(delay)
                elif error_code == '403':
                    raise PermissionError(f"Access denied to {s3_key}")
                else:
                    raise

        raise FileNotFoundError(f"❌ Upload verification failed after {retries} attempts: {s3_key}")