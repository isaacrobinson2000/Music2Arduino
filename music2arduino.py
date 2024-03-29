"""
Converts music files to Arduino Piezo buzzer code using music21 library
"""

import sys
from numbers import Integral
from typing import List, Union, Tuple, Optional

from music21.chord import Chord
from music21.converter import parse
from music21.stream import Stream
from music21.note import Rest, Note
from music21.stream import Score

# Used for test playing ardiuno melody results...
import pyaudio
import numpy as np

# Maximum number that can fit in a byte
BYTE_LIMIT = 255
# Notes to put per line in final output C++ code.
NOTES_PER_LINE = 5
# Select Slice style
CUT_EDGE_NOTES = True
# C++ Source Code for playing files via a piezo buzzer connected to a pin on the Arduino
ABOVE_NOTES_CODE = """
/* 
 * Plays an array of uint16 data generated using music2arduino.py does this by iterating, 
 * playing each note for each duration. Specifics on the format covered else where. 
 */

// The pin number for the piezo buzzer
const int PIEZO_PIN = 6;
"""
BELOW_NOTES_CODE = """
void setup() {
  // Nothing to do in here...
}

void loop() {
  // Wait 3 seconds, then play the song....
  delay(3000);
  play_song(PIEZO_PIN, tune1, tune1Length, tickMillis);
}

/**
 * Coverts a midi number to a frequency so a peizo buzzer can play it.
 * 
 * @param midi_num: A midi number to convert to a frequency, from 0 - 128
 * @returns: A integer representing the frequency of the sound in hertz.
 */
int to_frequency(uint8_t midi_num) {
  return (int)(pow(2, ((double)midi_num - 69) / 12) * 440);
}

/**
 * Plays a song through a peizo buzzer using the arduino.
 * 
 * @param piezo_pin: The pin the piezo buzzer is on.
 * @param song: The song, in the form of a array of 16 bit integers.
 * @param len: The length of the song array.
 * @param tick_speed: The milliseconds per duration tick to play notes at.
 */
void play_song(int piezo_pin, const uint16_t song[], const int len, double tick_speed) {
  for(int i=0; i < len; i++) {
    play_note_at(piezo_pin, i, song, tick_speed);
  }
}

/**
 * Plays a single note within a song. Used by play_song to play each note.
 * 
 * @param piezo_pin: The pin the piezo buzzer is on.
 * @param note_i: The index of the note within the song array to play.
 * @param song: The song, in the form of a array of 16 bit integers.
 * @param tick_speed: The milliseconds per duration tick to play notes at.
 */
void play_note_at(int piezo_pin, int note_i, const uint16_t song[], double tick_speed) {
  // Grab the note from flash memory...
  uint16_t data = (uint16_t *)pgm_read_word(&song[note_i]);
  // Split the note into its midi number and duration data
  uint8_t midi_num = data >> 8;
  uint8_t duration = (data << 8) >> 8;

  if((midi_num >> 7) == 1) {
    noTone(piezo_pin);
  }
  else {
    tone(piezo_pin, to_frequency(midi_num));
  }

  delay((unsigned long)(duration * abs(tick_speed)));
  noTone(piezo_pin);
}
"""


# CLASS USED FOR PLAYING NOTES
class NotePlayer:
    # Used for smoothing the ends of the sine waves to remove "clicking sounds" when transitioning between notes...
    SMOOTH_FRONT = np.arange(300) / 300
    SMOOTH_BACK = SMOOTH_FRONT[::-1]

    def __init__(self, sampling_rate: int = 44100, volume: float = 0.5):
        """
        Create a new note player object for playing raw frequencies...

        :param sampling_rate: The sampling rate in Hertz for this NotePlayer, defaults to 44100, must be an integer.
        :param volume: The volume to output notes at, a float between 0 and 1, 1 being the max volume and 0 being no
                       volume.
        """
        # Set the volume to play at
        self.volume = volume
        # Set the sampling rate for this NotePlayer....
        if(sampling_rate > 0):
            self._SAMPLING_RATE = int(sampling_rate)
        else:
            raise ValueError("Sampling rate must be greater then 0!!!")
        # Set the current time into the track to 0...
        self._time = 0

        # Create required pyaudio objects...
        self._py_audio = pyaudio.PyAudio()
        self._output_stream = self._py_audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate = int(self._SAMPLING_RATE),
            output=True
        )


    @property
    def volume(self):
        """
        The volume of this NotePlayer, a float between 0 and 1.
        """
        return self._volume

    @volume.setter
    def volume(self, value):
        """
        The volume of this NotePlayer, a float between 0 and 1.
        """
        if(0 <= value <= 1):
            self._volume = value
        else:
            raise ValueError(f"Volume can't be {value}, must be a float between 0 and 1...")

    def play(self, frequency: float, duration: float, volume: float = None):
        """
        Play a Note of a specified frequency and duration using pyaudio.

        :param frequency: The frequency, in hertz, of the tone.... a float...
        :param duration: The duration of the note, in seconds, as a float...
        :param volume: The volume to play back at, defaults to None. If it's default value, it will just use the global
                       volume value for this NotePlayer. Note, setting this volume does NOT change the global volume
                       attribute for this note player, only overrides is while playing this single note.
        :returns: Nothing, throws error if any problems occur durring playing
        """
        # Get correct volume value based on arguments passed...
        volume = self.volume if (volume is None or (not (0 <= volume <= 1))) else volume

        # Check the duration is greater then zero...
        if(duration <= 0):
            return

        # Grab the audio samples...
        samples = self._get_sine_samples(frequency, duration, volume)
        # Write the samples...
        self._output_stream.write(samples.tostring())

    def _get_sine_samples(self, frequency: float, duration: float, volume: float) -> np.ndarray:
        """
        Private Method: Gets the sine wave samples needed to play an audio tone for the given values.

        :param frequency: Frequency of the tone, in hertz.
        :param duration: Duration of the tone, in seconds.
        :param volume: Volume of the tone, as a value between 0 and 1
        :return: A numpy array of audio samples in float32 format...
        """
        sample_len = int(duration * self._SAMPLING_RATE)
        # Compute sampling factor... Effects how fast the sine wave oscillates...
        sample_factor = (float(frequency) * 2 * np.pi) / self._SAMPLING_RATE
        # Compute numpy array of audio sample for this tone, using the sine wave...
        samples = (float(volume) * np.sin(np.arange(sample_len) * sample_factor)).astype(np.float32)
        # If note is shorter then 600 samples, shorten front and back to fit the sample
        if(len(samples) < 600):
            smooth_front = self.SMOOTH_FRONT[:int(len(samples) // 2)]
            smooth_back = self.SMOOTH_BACK[-int(len(samples) // 2):]
        else:
            smooth_front = self.SMOOTH_FRONT
            smooth_back = self.SMOOTH_BACK

        # Smooth the tips of the sample...
        samples[:len(smooth_front)] = samples[:len(smooth_front)] * smooth_front
        samples[-len(smooth_back):] = samples[-len(smooth_back):] * smooth_back
        # Increment the current time...
        self._time += duration

        # Return the audio samples
        return samples


    def __del__(self):
        """
        Deletes pyaudio object created at object creation.
        """
        self._output_stream.stop_stream()
        self._output_stream.close()
        self._py_audio.terminate()


# Represents a simple Note
class MidiNote:
    """
    Represents a simple midi note. Includes the start location, duration, and midi frequency value.
    """
    def __init__(self, midi_num: int = 0, start: Integral = 0, duration: Integral = 1):
        """
        Create a new MidiNote...

        :param midi_num: The midi number of this note, an integer.
        :param start: The starting offset of this note measured in quarter note lengths, a Fraction/Float.
        :param duration: The duration of this note measured in quarter note lengths, a Fraction/Float.
        """
        self._midi_num = None
        self._start = None
        self._duration = None

        self.midi_num = midi_num
        self.start = start
        self.duration = duration


    @property
    def midi_num(self):
        """
        Midi number for this note, a number between 0 and 127, 0 signifying a rest...
        """
        return self._midi_num

    @midi_num.setter
    def midi_num(self, value: int):
        """
        Midi number for this note, a number between 0 and 127, 0 signifying a rest...
        """
        if(value is None):
            self._midi_num = 0
        else:
            if (value < 0 or value > 127):
                raise ValueError("Midi Number can't be negative or greater then 127...")

            self._midi_num = value


    @property
    def start(self):
        """
        The starting point of this note relative to the global list this is part of, in quarters notes.
        """
        return self._start

    @start.setter
    def start(self, value: Integral):
        """
        The starting point of this note relative to the global list this is part of, in quarters notes.
        """
        self._start = value

    @property
    def duration(self):
        """
        The duration of the note, in quarter notes.
        """
        return self._duration

    @duration.setter
    def duration(self, value: Integral):
        """
        The duration of the note, in quarter notes.
        """
        if(value <= 0):
            raise ValueError("Note must have a duration greater then 0!!!")

        self._duration = value


    @property
    def end(self):
        """
        The end of this note from the start of the list of notes...
        """
        return self.start + self.duration

    @end.setter
    def end(self, value: Integral):
        """
        The end of this note from the start of the list of notes...
        """
        self.duration = value - self.start

    def __str__(self):
        return f"MidiNote(tone: {self.midi_num}, start: {self.start}, duration: {self.duration})"


# Represents changes in tempo...
class TempoChange:
    """
    Represents a tempo change within a piece. Stores the speed in seconds per quarter note...
    """
    # The default seconds per quarter note in a midi file....
    DEFAULT_VAL = 0.5

    def __init__(self, quat_per_second: float = None, start_offset = None):
        """
        Creates a new TempoChange object.

        :param quat_per_second: A float which is the seconds per quarter note for this tempo change object. Defaults
                                to 0.5 seconds per quarter note if not set or set to none.
        :param start_offset: The starting offset location of this tempo change object.
        """
        self._secs_per_q = None
        self.seconds_per_quarter = quat_per_second
        self.start_offset = 0 if (start_offset is None) else start_offset

    @property
    def seconds_per_quarter(self) -> float:
        """
        The seconds per quarter note for this tempo change object...
        """
        return self._secs_per_q

    @seconds_per_quarter.setter
    def seconds_per_quarter(self, val: float):
        """
        The seconds per quarter note for this tempo change object...
        """
        if(val is None):
            self._secs_per_q = self.DEFAULT_VAL
        elif(val <= 0):
            raise ValueError("Can't have a seconds per quarter value less then or equal to 0!")

        self._secs_per_q = val




# UTILITY METHODS
def note_midi(note: Union[Note, Rest, Chord]) -> Union[int, None]:
    """
    Gets the midi number for this note by taking the chords root, or returns none if it is a rest

    :param note: A music21 General note to get the pitch from
    :return: A integer if note has a pitch, otherwise None
    """
    if(note.isRest):
        return None
    elif(note.isChord):
        return note.root().midi
    else:
        return note.pitch.midi


def to_note_list(music: Score, track_priorities: Optional[List[int]], min_gap: Integral = 2) -> List[MidiNote]:
    """
    Processes a music21 Stream, representing a piece of music, and converts it into a list of MidiNotes which can be
    turned into arduino code easily...

    :param music: A music21 music score, including all of the tracks or parts of the music...
    :param track_priorities: A list of integers specifying to order to insert seprate track notes into the main track.
    :param min_gap: An integral being the minimum gap of a rest or empty spot to allow for insertions. Defualts to
                    2, or the length of a half note...
    :return: A list of MidiNotes, being the simplified version of the song passed to the method.
    """
    if(track_priorities is None):
        track_priorities = list(range(len(music.parts)))

    # Create the track stack...
    print("Chordifying Stuff...")
    music_list = [track.flat.chordify().sorted for track in [music.parts[i] for i in track_priorities]]
    print("Starting My Stuff...")

    for i in range(len(music_list)):
        music_list[i] = [MidiNote(note_midi(notey), music_list[i].elementOffset(notey), notey.duration.quarterLength) for notey in music_list[i].notesAndRests]

    # Get the min and max offsets for all stream parts, and use it to add rests to the ends
    min_offset = min([track[0].start for track in music_list if (len(track) > 0)])
    max_offset = max([track[-1].end for track in music_list if (len(track) > 0)])

    # Rest adding logic for edges...
    for track in music_list:
        track: List[MidiNote] = track
        if(track[0].start > min_offset):
            track.insert(0, MidiNote(0, min_offset, track[0].start - min_offset))
        if(track[-1].end < max_offset):
            track.append(MidiNote(0, track[-1].end, max_offset - track[-1].end))

    # Grab the start and end points...
    start = min([track[0].start for track in music_list])
    end = max([track[-1].end for track in music_list])

    # Execute helper...
    new_list = []

    _to_note_list(music_list, new_list, start, end, min_gap)

    # Return the new list...
    return new_list


def _to_note_list(music_stack: List[List[MidiNote]], new_list: List[MidiNote], start_point: Integral, end_point: Integral, min_gap: Integral):
    """
    Recursive helper for to_note_list... performs brunt of track merging... Stores results in new_list.

    :param music_stack: Stack of note lists to get notes from by recurively inserting into empty gaps(rests)...
    :param new_list: The list that the merged results will be stored in...
    :param start_point: Time starting point...
    :param end_point: Time ending point...
    :param min_gap: The minimum gap to allow for note insertion, defaults to 2, or the length of the half note...
    """
    # Check if this gap is above the minimum threshold...
    if((end_point - start_point) <= min_gap):
        new_list.append(MidiNote(0, start_point, end_point - start_point))
        return

    # Get starting index...
    i = _get_starting_index(music_stack[0], start_point)
    track = music_stack[0]

    # Flag loop used to stop loop after finding the last note
    end_not_found = True

    while(end_not_found):
        # Code here
        temp_note = MidiNote(track[i].midi_num, track[i].start, track[i].duration)

        # If on one of the end bounding notes, shorten the notes to fit within the bounds:
        # Start note bound...
        if(temp_note.start < start_point):
            end_temp = temp_note.end
            temp_note.start = start_point
            temp_note.end = end_temp

        # End note bound...
        if(temp_note.end >= end_point):
            temp_note.end = end_point
            # End has been found, set end flag to end the loop...
            end_not_found = False

        # If the note is a rest, and there is another track on the stack, perform the recursive call...
        if(temp_note.midi_num == 0 and len(music_stack) > 1):
            _to_note_list(music_stack[1:], new_list, temp_note.start, temp_note.end, min_gap)
        # Otherwise, add this note
        else:
            new_list.append(temp_note)

        # Increment index...
        i += 1

# TODO: Write time search, should be even faster then binary search
#  (desired time <> current time <> index, start at end...)
def _get_starting_index(track: List[MidiNote], time: Integral) -> int:
    """
    Performs a binary search to find the first note overlapping a given time...

    :param track: The track to find the time stamp within, a list of MidiNote.
    :param time: An integral, the time to find where the note lands within.
    :return: An integer, being the index of the note that first overlaps the given time stamp...
    """
    high = len(track) - 1
    low = 0
    mid = (high + low) // 2

    # While the time marker is not between the note's endpoints....
    while(not (track[mid].start <= time < track[mid].end)):
        # If the start of the note is greater, move to lower half...
        if(track[mid].start > time):
            high = mid - 1
            mid = (high + low) // 2
        # If the end of the note is less, move to the upper half
        elif(track[mid].end <= time):
            low = mid + 1
            mid = (high + low) // 2
        else:
            raise ArithmeticError("Uhm... Should never get here...")

    return mid


def to_8_bits(value: int) -> str:
    """
    Returns a string representing the 8-bit binary form of this number

    :param value: A positive integer/number, 0 - 255 recommended. If the number is greater then 255, it will be wrapped
                  around to fit into a 8-bit representation. Negative values are made positive.
    :return: String, of 8-bit format (Ex. "10100100")
    """
    value = abs(value) % 256

    return f"{value:08b}"


def apply_tempos(track: List[MidiNote], track_speeds: List[TempoChange]) -> List[Tuple[Optional[int], float, float]]:
    """
    Apply tempos to the passed track list, returning a list of notes with durations and starting points in seconds.

    :param track: The track to apply tempos to, a list of MidiNote.
    :param track_speeds: The list of tempos for the track, a list of TempoChange.
    :return: A list of length 3 tuples, containing:
                - An optional integer being the midi tone number, or None if this is a rest.
                - A float being the duration of the note in seconds.
                - A float being the offset of the note in seconds.
    """
    # Iter 1: Get the longest note in the list, and convert all durations to seconds....
    current_track_speed = 0
    notes_in_seconds = []
    track_location = 0

    for i, notey in enumerate(track):
        # If we move into the interval of next tempo change object, move to the next tempo change object...
        if(current_track_speed + 1 < len(track_speeds)):
            if(notey.start >= track_speeds[current_track_speed + 1].start_offset):
                current_track_speed += 1

        if(i == 0):
            track_location = notey.start * track_speeds[current_track_speed].seconds_per_quarter

        duration = notey.duration * track_speeds[current_track_speed].seconds_per_quarter
        notes_in_seconds.append((notey.midi_num, duration,
                                 track_location))
        track_location += duration

    return notes_in_seconds


def scale_durations(track_seconds: List[Tuple[Optional[int], float, float]]) -> Tuple[float, List[Tuple[Optional[int], int]]]:
    """
    Scales the durations of the note list in-place in order to fit within the specified integer limit. It does this by
    setting the limit equal to the longest duration in the note list and scaling all of the rest of the notes to
    match the scale between the longest duration and the limit.

    :param track_seconds: A list of midi notes, being in seconds as from the apply_tempos method.

    :return: Two (or three) items in a tuple:
                - A float being the milliseconds per tick to play the song at a normal speed.
                - A list of tuples with two integers, first integer is midi tone number, second integer is the duration
                  from 1-255.
    """
    # Computing minimum seconds spent by a single note...
    max_secs = max(duration for (tone, duration, offset) in track_seconds)

    # Compute the tick speed for the song... Since the longest note gets a value of 255 ticks...
    tick_speed = (max_secs / BYTE_LIMIT) * 1000

    # List to store final results...
    scaled_durations: List[Tuple[Optional[int], int]] = []

    # Now Iter 2... Scale the durations
    for tone, duration, offset in track_seconds:
        scaled_durations.append((None if(tone == 0) else tone, int((duration / max_secs) * BYTE_LIMIT)))

    return tick_speed, scaled_durations


def scale_durations_multitrack(track_seconds: List[Tuple[str, float, float]]) -> Tuple[float, List[Tuple[str, int, int]]]:
    """
    Scales the durations of the note list in-place in order to fit within the specified integer limit. It does this by
    setting the limit equal to the longest duration in the note list and scaling all of the rest of the notes to
    match the scale between the longest duration and the limit. This is the multi-track version.

    :param track_seconds: A list of midi notes, being in seconds as from the multi-track method.

    :return: Two (or three) items in a tuple:
                - A float being the milliseconds per tick to play the song at a normal speed.
                - A list of tuples with a string being the note type, and 2 integers, being the tone, and then duration
                  of the note scaled...
    """
    # (n_type, tone, dur)
    max_secs = max(dur for (n_type, tone, dur) in track_seconds)

    tick_speed = (max_secs / BYTE_LIMIT) * 1000

    scaled_durations: List[Tuple[str, int, int]] = []

    for n_type, tone, dur in track_seconds:
        scaled_durations.append((n_type, int(tone), int((dur / max_secs) * BYTE_LIMIT)))

    return (tick_speed, scaled_durations)


def get_tempo_list(stream: Score) -> List[TempoChange]:
    """
    Get the list of tempo changes for this musical score.

    :param stream: The music21 score to get tempos of.
    :return: A list of TempoChange objects, representing locations where the tempo has changed in the musical score.
    """
    return [TempoChange(obj.secondsPerQuarter(), start) for (start, end, obj) in stream.metronomeMarkBoundaries()]


def stream_to_notes(stream: Score, priority: List[int] = None) -> Tuple[float, List[Tuple[Optional[int], int]]]:
    """
    Converts a music21 stream of music to a list of notes and their durations

    :param stream: A music21 score containing parts(kind of like tracks)
    :param priority: A list of integers setting the priorities to put the streams in.
    :return: Two items in a tuple:
                - A float being the milliseconds per tick to play the song at a normal speed.
                - A list of tuples with two integers, first integer is midi tone number, second integer is the duration
                  from 1-255.
    """
    # Gather all tempo changes through out the song...
    print(stream.metronomeMarkBoundaries())

    # Convert to note list -> scale the durations -> then return the results...
    return scale_durations(apply_tempos(to_note_list(stream, priority), get_tempo_list(stream)))


def notes_to_arduino(noteys: List[Tuple[Optional[int], int]]) -> List[str]:
    """
    Converts a note list to a list of C++ byte sequences.

    :param noteys: The list of notes, each note actually being a tuple of 2 integers, where the first one is the
                   note's frequency midi number, and the second is the note's duration from 1-255. If the note is a
                   rest, the pitch midi number will be None.
    :return: A list of strings, being C++ 16-bit integers written in binary (Ex. '0b0100101101010010')
    """
    byte_list = []

    for tone, dur in noteys:
        note_byte = to_8_bits(tone) if (tone is not None) else "10000000"

        byte_list.append("0b" + note_byte + to_8_bits(dur))

    return byte_list


# MAIN METHODS

def main(args: List[str]):
    """
    Default main method, converts music file to arduino peizo buzzer code and prints it to the console

    :param args: Args this main method accepts, this one accepts file name, followed by a list of integers separated
                 by spaces representing the priority to insert tracks in...
    :return: Nothing
    """
    if(len(args) > 0):
        # Parsing arguments to get priority list and file to convert
        streamer = parse(args[0])
        args = args[1:]

        priority_list = [int(i) for i in args]
        if(len(priority_list) == 0):
            priority_list = None

        # Begin processing
        tick_speed, byte_list = stream_to_notes(streamer, priority_list)
        byte_list = notes_to_arduino(byte_list)

        # Begin print source code
        print("\n\nCopy and Paste the Code Below: \n")
        print(ABOVE_NOTES_CODE)
        print("const uint16_t tune1[] PROGMEM = {", end='')

        for i, uint16 in enumerate(byte_list):
            if(i % NOTES_PER_LINE == 0):
                print()

            ender = ", " if (i < len(byte_list) - 1) else "\n"

            print(uint16, end=ender)

        print("};")
        print("const unsigned int tune1Length = sizeof(tune1) / sizeof(uint16_t);")
        print(f"const double tickMillis = {tick_speed};  // Adjust to speed up and slow down the song...")
        print(BELOW_NOTES_CODE)


def note_cut(streamer: Stream, amount: float) -> Stream:
    """
    Cuts all notes in this stream such that all note longer then the set

    :param streamer: # TODO DOCUMENTATION
    :param amount:
    :return:
    """
    for notey in streamer.notesAndRests:
        if((not notey.isRest) and notey.duration.quarterLength > amount):
            notey.duration.quarterLength = amount

    return streamer


def debug_main(args: List[str]):
    """
    Debug main method, converts music file to a arduino note list, prints them in a human readable way, and then prints
    tracks contained in this music file and the instruments attached to each one.

    :param args: Args this main method accepts, this one accepts file name, followed by a list of integers separated
                 by spaces representing the priority to insert tracks in...
    :return: Nothing
    """
    if(len(args) > 0):
        # Parsing arguments to get priority list and file to convert
        streamer: Score = parse(args[0])
        args = args[1:]

        priority_list = [int(i) for i in args]
        if(len(priority_list) == 0):
            priority_list = None

        # Begin processing
        tick_speed, note_list = stream_to_notes(streamer, priority_list)

        # Print notes to screen
        for tone, dur in note_list:
            print(f"Midi Number: {tone}, Duration: {dur}")

        for i, p in enumerate(streamer.parts):
            print(f"Track Number: {i}, Instrument: {p.getInstrument()}")

        print(f"Milliseconds per tick: {tick_speed}")


SINE_WAVE_LENGTH = 300
SAMPLES_PER_LINE = 10

def multi_track_main(args: List[str]):
    if(len(args) > 0):
        import heapq
        # Parsing arguments to get priority list and file to convert
        streamer: Score = parse(args[0])
        args = args[1:]

        if(len(args) == 0):
            priority_lists = [[num] for num in list(range(len(streamer.parts)))]
        else:
            priority_lists = [[int(num.strip()) for num in arg.split(",")] for arg in args]

        print(f"Number of tracks: {len(priority_lists)}")

        # Begin processing
        t = get_tempo_list(streamer)
        # We reverse tuples so track offset is considered first.
        results = [apply_tempos(to_note_list(streamer, priority_list), t) for priority_list in priority_lists]
        # We reverse all tones and also remove all rests since we must execute a clear after every note...
        results = [[n[::-1] for n in lst if((n[0] is not None) and (n[0] != 0))] for lst in results if(len(lst) > 0)]

        starts_and_ends = [[] for i in range(len(results))]

        for i, lst in enumerate(results):
            for off, dur, tone in lst:
                starts_and_ends[i].append((off, tone, "start"))
                starts_and_ends[i].append((off + dur, tone, "end"))

        # Custom merge algorithm: We need to merge notes based on which has the smaller offset...
        # This can be done efficiently with a heap, which python has.
        heap = []
        offset = [0 for i in range(len(starts_and_ends))]
        merged = []

        for i, lst in enumerate(starts_and_ends):
            heapq.heappush(heap, (lst[0], i))

        while(len(heap) > 0):
            next_note, idx = heapq.heappop(heap)
            merged.append(next_note)
            offset[idx] += 1
            if(offset[idx] < len(starts_and_ends[idx])):
                heapq.heappush(heap, (starts_and_ends[idx][offset[idx]], idx))

        final_note_instructions = []

        for i, (off, tone, n_type) in enumerate(merged):
            if(i < (len(merged) - 1)):
                final_note_instructions.append((n_type, tone, merged[i + 1][0] - off))
            else:
                final_note_instructions.append((n_type, tone, 0))

        print("\n\nCopy and Paste the Code Below: \n")

        # Generate and dump sine wave based on track count....
        print("const PROGMEM uint8_t sineWave[] = {", end='')

        sine_wave = (((np.sin(np.linspace(0, 2 * np.pi, SINE_WAVE_LENGTH)) + 1) / 2) * (255 / len(priority_lists))).astype(np.uint8)

        for i, val in enumerate(sine_wave):
            if(i % SAMPLES_PER_LINE == 0):
                print("\n  ", end='')
            print(val, end=", " if (i < len(sine_wave) - 1) else "\n};\n")

        print("const size_t sineWaveLen = sizeof(sineWave) / sizeof(uint8_t);\n\n")

        print("const uint16_t PROGMEM music[] = {", end='')

        tick_speed, final_note_instructions = scale_durations_multitrack(final_note_instructions)

        for i, (n_type, tone, dur) in enumerate(final_note_instructions):
            if(i % NOTES_PER_LINE == 0):
                print("\n  ", end='')

            ender = ", " if (i < len(final_note_instructions) - 1) else "\n"

            print(f"0b{1 * (n_type == 'end')}{to_8_bits(tone)[1:]}{to_8_bits(dur)}", end=ender)

        print("};")
        print("const size_t musicLen = sizeof(music) / sizeof(uint16_t);")
        print(f"const float millisPerTick = {tick_speed};")


def play_main(args: List[str]):
    """
    Play main method, converts music file to a arduino byte list, and then plays it as it would be played on the
    arduino. Prints some debug info while doing so.

    :param args: Args this main method accepts, this one accepts file name, followed by a single integer representing
                 the milliseconds per tick, which defaults to 30, followed by a list of integers separated
                 by spaces representing the priority to insert tracks in...
    :return: Nothing
    """
    if(len(args) > 0):
        # Parsing arguments to get priority list and file to convert
        streamer: Score = parse(args[0])
        args = args[1:]

        # Accepts one extra arg, the ticks, or milliseconds per tick, defaults to 30 just like arduino code
        ticks = None

        if(len(args) > 0 and args[0].isdigit()):
            ticks = int(args[0])
            args = args[1:]

        priority_list = [int(i) for i in args]
        if (len(priority_list) == 0):
            priority_list = None

        # Get the byte list
        normal_ticks, data = stream_to_notes(streamer, priority_list)
        ticks = normal_ticks if(ticks is None or (ticks <= 0)) else ticks
        byte_list = notes_to_arduino(data)

        # Mimics Arduino output as closely as possible... By just using sine waves to represent notes...
        note_player = NotePlayer()
        note_player.volume = 0.5

        # Function to convert midi numbers to frequencies
        def to_freq(m: int) -> float:
            return (2 ** ((m - 69) / 12)) * 440

        try:
            # Attempt to play sound using pysine
            for uint16 in byte_list:
                # Convert C++ byte strings to python integers
                b = int(uint16[2:], base=2)

                # Grab first byte, it is the duration
                dur = b & 0b11111111
                # Grab the frequency, the last byte of 16 bit integer
                freq = (b >> 8)

                if (freq >> 7 == 1):
                    # If the very first bit of the frequency is 1, this is suppose to be a rest, so sleep for the
                    # duration
                    print(f"Rest, Duration: {(dur * ticks) / 1000}sec")
                    note_player.play(0, (dur * ticks) / 1000, 0)
                else:
                    # Otherwise, play the frequency for the specified duration
                    print(f"Frequency: {to_freq(freq):.02f}, Duration: {(dur * ticks) / 1000:.02f}sec")
                    note_player.play(to_freq(freq), (dur * ticks) / 1000)
        except KeyboardInterrupt:
            # If user presses CTRL+C, which is common, display cleaner error message since this isn't really an error
            # but the intended way to cancel the playing.
            print("Keyboard Interrupt...")
        finally:
            print("Shutting down sound system...")
            del note_player

# TODO: BETTER CLI API
if(__name__ == '__main__'):
    print(list(sys.argv))
    if(len(sys.argv) >= 2 and sys.argv[1] == "debug"):
        debug_main(sys.argv[2:])
    elif(len(sys.argv) >= 2 and sys.argv[1] == "multitrack"):
        multi_track_main(sys.argv[2:])
    elif(len(sys.argv) >= 2 and sys.argv[1] == "play"):
        play_main(sys.argv[2:])
    else:
        main(sys.argv[1:])
