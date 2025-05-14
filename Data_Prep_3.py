import os
import pandas as pd
import pretty_midi
import bisect
import xml.etree.ElementTree as ET

ASAP_ROOT = '/Users/arminhamrah/Downloads/MIR/asap'
FEATURES_ROOT = '/Users/arminhamrah/Downloads/MIR/asap_features'

# ======================

def load_downbeats(path):
    """
    Read annotations file and return sorted list of downbeat times (seconds),
    always including 0.0 as first downbeat.
    """
    times = [0.0]
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and 'db' in parts[2]:
                try:
                    times.append(float(parts[0]))
                except ValueError:
                    pass
    return sorted(times)


def extract_time_signatures(xml_file_path):
    """
    Parse MusicXML to extract time signatures per measure.
    Returns a sorted list of tuples: (measure_number, "beats/beat_type").
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    ts_points = []
    for measure in root.findall('.//measure'):
        num = measure.attrib.get('number')
        time_elem = measure.find('attributes/time')
        if time_elem is not None and num is not None:
            beats = time_elem.find('beats').text
            beat_type = time_elem.find('beat-type').text
            try:
                m = int(num)
                ts_points.append((m, f"{int(beats)}/{int(beat_type)}"))
            except (ValueError, TypeError):
                continue
    # keep only first occurrence of each signature change
    # sorted by measure
    unique = {}
    for m, sig in ts_points:
        if m not in unique:
            unique[m] = sig
    sorted_points = sorted(unique.items())
    return sorted_points


def extract_notes(midi_path, annotation_path, ts_points):
    """
    Extract per-note data from a MIDI file using downbeat annotations and normalize measures.
    Returns a DataFrame with columns: measure, note, onset, offset, measure_start, measure_end
    """
    downbeats = load_downbeats(annotation_path)
    pm = pretty_midi.PrettyMIDI(midi_path)

    # computes end time for each measure
    intervals = [downbeats[i+1] - downbeats[i] for i in range(len(downbeats)-1)]
    last_interval = intervals[-1] if intervals else 0.0
    end_times = downbeats[1:] + [downbeats[-1] + last_interval]

    # Build a dict of normalized durations based on time signature
    def sig_to_duration(ts):
        if ts is None:
            return 2.0  # default to 4/4 = 2 seconds
        try:
            beats, beat_type = map(int, ts.split("/"))
            return (60 / 120) * beats * (4 / beat_type)  # normalized to 120 BPM
        except:
            return 2.0

    sig_map = {}
    last_sig = "4/4"
    for i in range(len(downbeats)):
        measure_num = i + 1
        for m, s in reversed(ts_points):
            if measure_num >= m:
                last_sig = s
                break
        sig_map[measure_num] = sig_to_duration(last_sig)

    records = []
    for inst in pm.instruments:
        for note in inst.notes:
            onset, offset = note.start, note.end
            idx = bisect.bisect_right(downbeats, onset) - 1
            if idx < 0 or idx >= len(end_times):
                continue
            real_start, real_end = downbeats[idx], end_times[idx]
            real_duration = real_end - real_start

            norm_duration = sig_map.get(idx + 1, 2.0)
            rel_onset = (onset - real_start) / real_duration
            rel_offset = (offset - real_start) / real_duration
            norm_onset = rel_onset * norm_duration
            norm_offset = rel_offset * norm_duration

            measure_num = idx + 1
            measure_start = sum(sig_map.get(m, 2.0) for m in range(1, measure_num))
            measure_end = measure_start + norm_duration

            records.append({
                'measure': measure_num,
                'note': pretty_midi.note_number_to_name(note.pitch),
                'onset': round(norm_onset + measure_start, 5),
                'offset': round(norm_offset + measure_start, 5),
                'measure_start': round(measure_start, 5),
                'measure_end': round(measure_end, 5)
})
            
    return pd.DataFrame(records).sort_values(by='onset').reset_index(drop=True)

def assign_time_signature(df, ts_points):
    """
    Given a DataFrame with a 'measure' column and a sorted list of
    (measure_number, signature) tuples, assign a 'time_signature' column
    with the signature effective for that measure.
    """
    # if there are no time signatures found, default to None
    if not ts_points:
        df['time_signature'] = None
        return df

    def get_sig(m):
        # find the last signature change at or before measure m
        sig = ts_points[0][1]
        for m0, s in ts_points:
            if m >= m0:
                sig = s
            else:
                break
        return sig

    df['time_signature'] = df['measure'].apply(get_sig)
    return df


def main():
    # Walk through ASAP_ROOT
    for root, _, files in os.walk(ASAP_ROOT):
        rel = os.path.relpath(root, ASAP_ROOT)
        out_dir = os.path.join(FEATURES_ROOT, rel)
        os.makedirs(out_dir, exist_ok=True)

        # Load time signatures once per piece
        xml_file = os.path.join(root, 'xml_score.musicxml')
        ts_points = []
        if os.path.isfile(xml_file):
            try:
                ts_points = extract_time_signatures(xml_file)
            except Exception as e:
                print(f"Error parsing XML {xml_file}: {e}")

        # Process each MIDI file
        for fname in files:
            if not fname.lower().endswith('.mid'):
                continue
            base = os.path.splitext(fname)[0]
            midi_path = os.path.join(root, fname)
            ann_path = os.path.join(root, f"{base}_annotations.txt")
            if not os.path.isfile(ann_path):
                print(f"Skipping {midi_path}: no annotation file found")
                continue

            try:
                df = extract_notes(midi_path, ann_path, ts_points)
                # Add time signature column
                df = assign_time_signature(df, ts_points)
                out_csv = os.path.join(out_dir, f"{base}.csv")
                df.to_csv(out_csv, index=False)
                print(f"Wrote {out_csv} ({len(df)} notes)")
            except Exception as e:
                print(f"Error processing {midi_path}: {e}")

if __name__ == '__main__':
    main()
