# -------------------------------------------
# THIS CLASS CANNOT BEING USED BECAUSE 
# THE UNDERLYING LOGIC MAKE THIS CANNOT BE USED
# EVEN THIS WAS OVERRIDEN THE PARRENT METHOD
# ---------------------------------------------- 
from fastrtc import ReplyOnPause
import numpy as np
import json


class ReplyAfterSilence(ReplyOnPause):
    def __init__(self, 
                 fn, 
                 startup_fn = None, 
                 algo_options = None, 
                 model_options = None, 
                 can_interrupt = True, 
                 expected_layout = "mono", 
                 output_sample_rate = 24000, 
                 output_frame_size = None, 
                 input_sample_rate = 48000, 
                 model = None, 
                 needs_args = False,
                 *,
                 silence_timeout_s: float = 10.0):
        super().__init__(fn, startup_fn, algo_options, model_options, can_interrupt, expected_layout, output_sample_rate, output_frame_size, input_sample_rate, model, needs_args)
        self.silence_timeout_s = float(silence_timeout_s)

        if not hasattr(self.state, "silence_chunks"):
            self.state.silence_chunks = 0

    def determine_pause(
        self, audio: np.ndarray, sampling_rate: int, state
    ) -> bool:
        """
        Analyzes an audio chunk to detect if a significant pause occurred after speech.

        Uses the VAD model to measure speech duration within the chunk. Updates the
        application state (`state`) regarding whether talking has started and
        accumulates speech segments.

        Args:
            audio: The numpy array containing the audio chunk.
            sampling_rate: The sample rate of the audio chunk.
            state: The current application state.

        Returns:
            True if a pause satisfying the configured thresholds is detected
            after speech has started, False otherwise.
        """
        print(">>> OVERRIDE CONFIRMED:", self.__class__.__name__)
        duration = len(audio) / sampling_rate
        print(f"duration:{duration}")
        if duration >= self.algo_options.audio_chunk_duration:
            dur_vad, _ = self.model.vad((sampling_rate, audio), self.model_options)
            print(f"dur_vad:{dur_vad}")
            if (
                dur_vad > self.algo_options.started_talking_threshold
                and not state.started_talking
            ):
                state.started_talking = True
                print('Started talking')
                self.send_message_sync(json.dumps({"type": "log", "data": "started_talking"}))
            if state.started_talking:
                if state.stream is None:
                    state.stream = audio
                else:
                    state.stream = np.concatenate((state.stream, audio))

                # Check if continuous speech limit has been reached
                current_duration = len(state.stream) / sampling_rate
                print(f"current_duration:{current_duration}")
                if current_duration >= self.algo_options.max_continuous_speech_s:
                    print(f'max_continuous_speech_s:{self.algo_options.max_continuous_speech_s}')
                    return True
            state.buffer = None

            if state.started_talking:
                if dur_vad < self.algo_options.speech_threshold:
                    state.silence_chunks += 1
                else:
                    state.silence_chunks = 0  # reset when speech resumes

                silence_time = state.silence_chunks * self.algo_options.audio_chunk_duration
                if silence_time >= self.silence_timeout_s:
                    print(f'silence_time:{silence_time}')
                    print(f'silence_timeout_s:{self.silence_timeout_s}')
                    state.pause_detected = True
                    return True
                
        return False
