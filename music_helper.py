import subprocess
import sys
import os
import music21
import pygame
import time
from PIL import Image


# run_oemer_with_updates("Happy-Birthday.png")
# get_notes_from_xml("Happy-Birthday.musicxml")
# play_violin_mp3_library("Happy-Birthday.musicxml")
# generate_visual_sheet("Happy-Birthday.musicxml")
# update_musicxml(xml_path: str, corrected_notes_text: str)


def run_oemer_with_updates(image_path: str) -> str:
    message = ""
    print(f"🚀 Starting oemer for: {image_path}...")
    print("Note: This can take several minutes. Watching the logs below:\n")

    try:
        # We use Popen instead of run to stream the output line by line
        process = subprocess.Popen(
            ["oemer", image_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge error and output streams
            text=True,
            bufsize=1,
        )

        # Print the output as it happens
        for line in process.stdout:
            print(f"[oemer]: {line.strip()}")
            sys.stdout.flush()

        process.wait()

        if process.returncode == 0:
            message = "\n✅ Success! Check your folder for the MIDI/MusicXML files."
        else:
            message = f"\n❌ Oemer exited with error code: {process.returncode}"

    except FileNotFoundError:
        message = (
            "❌ Error: 'oemer' command not found. Is it installed in your environment?"
        )
    except Exception as e:
        message = f"❌ An unexpected error occurred: {e}"

    return message


def process_and_play(full_path: str) -> str:
    file_path = os.path.basename(full_path)
    print(file_path)
    try:
        generate_visual_sheet(file_path)
    except Exception as e:
        print(f"GUI Display failed ({e}), saving to file instead...")
        save_visual_sheet(file_path)
    try:
        play_violin_mp3_library(file_path)
    except Exception as e:
        print(f"Audio device failed ({e}), rendering MIDI instead...")
        save_violin_mp3_library(file_path)
    try:
        return get_notes_from_xml(file_path)
    except Exception as e:
        print(f"Failed ({e})...")


def get_notes_from_xml(file_path: str) -> str:
    message = ""
    notes_list = []
    if os.path.exists(file_path):
        message = f"🎉 Found the payload: {file_path}"

        # Now we 'call' the music21 tool to parse the results
        score = music21.converter.parse(file_path)

        # Print the first 10 notes found
        message = message + ("\n--- 🎼 NOTES DETECTED ---")
        notes = score.flatten().notes
        for i, n in enumerate(notes):
            if n.isNote:
                note_st = str(n.pitch.nameWithOctave + ":" + n.duration.type)
                notes_list.append(note_st)
            elif n.isChord:
                note_st = str(n.pitches + ":" + n.duration.type)
                notes_list.append(note_st)
        notes_str = ", ".join(notes_list)
        message = message + ":" + notes_str
    else:
        message = f"🤔 The tool finished, but I can't find '{file_path}' in the folder."
    return message


def save_violin_mp3_library(file_path):
    score = music21.converter.parse(file_path)
    midi_path = file_path.replace(".musicxml", ".mid")
    mp3_path = file_path.replace(".musicxml", ".mp3")

    # 1. Save as MIDI first
    score.write("midi", fp=midi_path)

    # 2. Convert MIDI to MP3 using fluidsynth (Common Linux tool)
    try:
        # This uses a soundfont to turn the 'instructions' into 'sound'
        # You'll need fluidsynth installed in your container
        subprocess.run(
            [
                "fluidsynth",
                "-ni",
                "/usr/share/sounds/sf2/FluidR3_GM.sf2",
                midi_path,
                "-F",
                mp3_path,
                "-r",
                "44100",
            ],
            check=True,
        )
        return f"MP3 created at {mp3_path}"
    except Exception as e:
        return f"MIDI created, but MP3 conversion failed: {e}"


def play_violin_mp3_library(
    xml_path: str, folder_name: str = "chromatic_samples_library"
) -> str:
    # Initialize the mixer for MP3 playback
    pygame.mixer.init()
    message = ""
    # 1. Load the MusicXML Tool
    if not os.path.exists(xml_path):
        message = f"🚨 XML file not found: {xml_path}"
        return message

    score = music21.converter.parse(xml_path)
    notes = score.flatten().notes

    message = f"🎻 Ready! Accessing library: {folder_name}..."

    for n in notes:
        if n.isNote:
            # 2. Get the Pitch Tag
            # MusicXML might give us 'G#' but your file is 'Ab'
            pitch = n.pitch

            # Standardizing the name to match your "Ab1" format
            if pitch.accidental and pitch.accidental.name == "sharp":
                # Convert G#4 to Ab4
                filename_base = pitch.getLowerEnharmonic().nameWithOctave
            else:
                filename_base = pitch.nameWithOctave

            # Ensure flats use 'b' (e.g., A-4 becomes Ab4)
            filename = filename_base.replace("-", "b") + ".mp3"
            file_path = os.path.join(folder_name, filename)

            # 3. Execution (The Tool Call)
            if os.path.exists(file_path):
                # Scale duration (Quarter note = ~0.6 seconds)
                duration = n.duration.quarterLength * 0.6
                message += f"🔊 Playing: {filename}"

                try:
                    sound = pygame.mixer.Sound(file_path)
                    sound.play()

                    # Sleep so the note has time to ring out
                    time.sleep(duration)
                    # Fade out prevents a "pop" sound between notes
                    sound.fadeout(150)
                except Exception as e:
                    message += f"🚨 Error playing {filename}: {e}"
            else:
                message += f"⚠️ Missing from library: {filename}"
                # Still sleep so the rhythm stays correct
                time.sleep(n.duration.quarterLength * 0.6)

    pygame.mixer.quit()
    message += "✅ Performance finished!"
    return message


def save_visual_sheet(xml_path):
    score = music21.converter.parse(xml_path)
    # This saves the image as 'Happy-Birthday.png' (or similar) in your folder
    output_png = xml_path.replace(".musicxml", ".png")
    score.write("musicxml.png", fp=output_png)
    print(f"Visualization saved to: {output_png}")
    return output_png


def generate_visual_sheet(
    xml_path: str, library_folder: str = "visual_notes_library"
) -> str:
    # 1. Prepare Master Canvas
    master_canvas = Image.new("RGBA", (2500, 3000), (255, 255, 255, 255))
    staff_lines = Image.open(f"{library_folder}/blank_staff.png").convert("RGBA")

    score = music21.converter.parse(xml_path)
    notes = score.flatten().notes

    x_cursor = 100
    y_origin_offset = 0
    BEAT_WIDTH = 150
    LINE_SPACING = 500

    accumulated_duration = 0.0
    MAX_DURATION_PER_LINE = 4.0

    master_canvas.alpha_composite(staff_lines, (0, y_origin_offset))

    for n in notes:
        if n.isNote:
            if accumulated_duration > MAX_DURATION_PER_LINE:
                x_cursor = 100
                y_origin_offset += LINE_SPACING
                accumulated_duration = 0.0
                master_canvas.alpha_composite(staff_lines, (0, y_origin_offset))

            # --- STEM DIRECTION LOGIC (The Flip) ---
            # B4 is 35. If 35 or higher, use the reversed sticker.
            if n.pitch.diatonicNoteNum >= 35:
                sticker_name = f"{n.duration.type}_rev.png"
            else:
                sticker_name = f"{n.duration.type}.png"

            try:
                sticker = Image.open(f"{library_folder}/{sticker_name}").convert("RGBA")
            except FileNotFoundError:
                # Fallback to standard if rev doesn't exist
                sticker = Image.open(f"{library_folder}/{n.duration.type}.png").convert(
                    "RGBA"
                )

            # 2. Y-Position (Pitch + Current Staff Offset)
            pitch_steps = n.pitch.diatonicNoteNum
            y_offset = (544 + y_origin_offset) - (pitch_steps * 15)

            # 3. Paste Note
            master_canvas.alpha_composite(sticker, (int(x_cursor), int(y_offset)))

            # 4. Movement
            duration_value = n.duration.quarterLength
            x_cursor += duration_value * BEAT_WIDTH * 2
            accumulated_duration += duration_value

    master_canvas.show()
    rendered_image_path = "rendered_" + xml_path.split(".")[0] + ".png"
    master_canvas.save(rendered_image_path)


def update_musicxml(xml_path: str, corrected_notes_text: str) -> str:
    """
    Parses a string like 'G4:1.0, A4:0.5' and overwrites the MusicXML.
    Saves as 'filename.musicxml'
    """
    duration_map = {
        "16th": 0.25,
        "eighth": 0.5,
        "quarter": 1.0,
        "half": 2.0,
        "whole": 4.0,
    }

    # 1. Create a new music stream
    new_stream = music21.stream.Stream()

    # 2. Parse the text "Note:Duration, Note:Duration"
    # Logic: Split by comma, then split by colon
    try:
        note_entries = [n.strip() for n in corrected_notes_text.split(",")]
        for entry in note_entries:
            pitch_name, dur_val = entry.split(":")

            if dur_val.strip().lower() in duration_map:
                dur_val = duration_map[dur_val.strip().lower()]
            else:
                dur_val = float(dur_val)

            # Create a music21 Note object
            n = music21.note.Note(pitch_name)
            n.duration.quarterLength = dur_val
            new_stream.append(n)

        # 3. Handle File Naming
        # base_name = os.path.basename(xml_path)

        output_path = xml_path  # base_name

        # 4. Write the file
        new_stream.write("musicxml", fp=output_path)
        return get_notes_from_xml(output_path)

    except Exception as e:
        return f"Error parsing notes: {str(e)}. Please use format 'Pitch:Duration'."
