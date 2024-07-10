import os
import shutil
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from moviepy.audio.AudioClip import AudioClip
from moviepy.editor import AudioFileClip

from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId


def max_signed_value_for_bytes(n):
    return (1 << (8 * n - 1)) - 1


def extract_audio_from_vrs(vrs_file: str, output_audio: str, log_folder=None):
    """Extract audio from a VRS file and save it as an audio file."""
    use_temp_folder = False
    if log_folder is None:
        use_temp_folder = True
        log_folder = tempfile.mkdtemp()
    elif not os.path.exists(log_folder):
        os.mkdir(log_folder)

    converter = Vrs2Mp3Converter(vrs_file)
    duration_ns = converter.duration_ns()
    duration_in_second = duration_ns * 1e-9
    temp_audio_file = os.path.join(log_folder, "audio.mp3")

    if converter.contain_audio():
        audio_writer_clip = AudioClip(
            converter.make_audio_data,
            duration=duration_in_second,
            fps=converter.audio_config.sample_rate,
        )
        audio_writer_clip.nchannels = converter.audio_config.num_channels
        audio_writer_clip.write_audiofile(
            temp_audio_file,
            fps=converter.audio_config.sample_rate,
            buffersize=converter.audio_buffersize(),
        )
        audio_writer_clip.close()
    
    shutil.move(temp_audio_file, output_audio)

    if use_temp_folder:
        shutil.rmtree(log_folder)
    else:
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)


class Vrs2Mp3Converter:
    def __init__(self, vrs_path: str):
        self.provider_ = data_provider.create_vrs_data_provider(vrs_path)
        if not self.provider_:
            raise ValueError(f"vrs file: '{vrs_path}' cannot be read")

        self.audio_streamid_ = StreamId("231-1")
        self.contain_audio_ = self.provider_.check_stream_is_active(self.audio_streamid_)
        
        if self.contain_audio_:
            self.audio_config = self.provider_.get_audio_configuration(self.audio_streamid_)
            self.audio_max_value_ = max_signed_value_for_bytes(4)
            self.start_timestamp_ns_ = self.provider_.get_first_time_ns(self.audio_streamid_, TimeDomain.RECORD_TIME)
            self.end_timestamp_ns_ = self.provider_.get_last_time_ns(self.audio_streamid_, TimeDomain.DEVICE_TIME)
        
    def contain_audio(self) -> bool:
        return self.contain_audio_

    def duration_ns(self):
        return self.end_timestamp_ns_ - self.start_timestamp_ns_

    def audio_buffersize(self):
        audio_data = self.provider_.get_audio_data_by_index(self.audio_streamid_, 1)
        return len(audio_data[1].capture_timestamps_ns)

    def make_audio_data(self, t) -> np.ndarray:
        if self.contain_audio_ is False:
            raise SystemExit("The VRS file does not contain audio.")

        if np.size(t) == 1:
            return 0

        query_timestamp_ns = t * 1e9
        vrs_timestamp_ns = int(self.start_timestamp_ns_ + query_timestamp_ns[0])

        audio_data_and_config = self.provider_.get_audio_data_by_time_ns(
            self.audio_streamid_,
            vrs_timestamp_ns,
            TimeDomain.RECORD_TIME,
            TimeQueryOptions.CLOSEST,
        )
        audio_data = np.array(audio_data_and_config[0].data)
        audio_data = audio_data.astype(np.float64) / self.audio_max_value_

        return audio_data


if __name__ == "__main__":
    import tomllib
    config = tomllib.load(open("config.toml", "rb"))

    vrsfile = config["aria_recordings"]["vrs"]
    provider = data_provider.create_vrs_data_provider(vrsfile)
    output_audio_path = "audio.mp3"  

    extract_audio_from_vrs(vrsfile, output_audio_path)