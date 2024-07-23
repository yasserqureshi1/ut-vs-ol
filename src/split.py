import numpy as np

def get_time(track):
    track = np.array(track)
    return track[-1, 2] - track[0, 2]


def split_tracks(tracks, target, trial, size, over_lap):
    def window(track, size, over_lap):
        # Initalise empty list for segments
        segments = []

        # Time thats passed
        cumulative_time = 0

        # Position index
        position_id = 1

        # Start position index
        start = 0 

        while position_id < len(track):
            # Get time passed between `start` index and `position_id` index
            cumulative_time = get_time(track[start:position_id])

            # If time passed is greater than or equal to the segment size, and the time passed is less than 2*overlap
            if (cumulative_time >= size) and (cumulative_time < size*2):

                # Store as segment
                segments.append(track[start:position_id])

                # To calculate overlap...
                # Traverse backwards from `position_id`
                overlap_index = position_id - 1

                # Time thats passed for overlap
                cumulative_overlap = 0

                # Once time passed is greater than or equal to overlap size, or if at start index, break
                while cumulative_overlap < over_lap:
                    if overlap_index == 1:
                        break
                    cumulative_overlap = get_time(track[overlap_index:position_id])
                    overlap_index -= 1

                # That is now the index of the start
                start = overlap_index

                # Reset time passed value for next track
                cumulative_time = 0

            # If the time passed is less thatn 2*overlap, then use that as start position
            elif (cumulative_time > size*2):
                start = position_id
                
            position_id += 1
        return segments

    split_track = []
    split_target = []
    split_trial = []
    split_group = []
    track_id = 0
    while track_id < len(tracks):
        track_time = get_time(tracks[track_id])
        if track_time >= size:
            tk = window(tracks[track_id], size, over_lap)
            if tk == []:
                pass
            else:
                split_track += tk
                split_target += [target[track_id] for _ in range(len(tk))]
                split_trial += [trial[track_id] for _ in range(len(tk))]
                split_group += [track_id for _ in range(len(tk))]
        track_id += 1
    
    print('COMPLETED SPLIT TRACKS: ', len(split_trial))
    return np.array(split_track, dtype=object), np.array(split_target), np.array(split_trial), np.array(split_group)


def find_lowest_frame_for_trial(tracks, trials):
    unique_trials = np.unique(trials)
    frames = dict()
    for t in unique_trials:
        select = np.where(trials == t)[0]
        #print('SELECTION: ',t, select)
        #print(tracks[select[1]])
        lowest = tracks[select[1]][0,16]

        for track in tracks[select]:
            try:
                small = min(track[:,16])
                if small < lowest:
                    lowest = small
            except:
                pass
        
        frames[f'{t}'] = lowest
    return frames


def trial_duration_filter(tracks, targets, trials, min_time, max_time):
    print('Trial Duration Filter')
    print('Initial track num: ', len(tracks))
    split_track = []
    split_target = []
    split_trial = []
    track_id = 0
    frames = find_lowest_frame_for_trial(tracks, trials)
    while track_id < len(tracks):
        #print('Trial duration filter: ', track_id)
        lowest = frames[f'{trials[track_id]}']
        track = []
        for i, pos in enumerate(tracks[track_id]):
            p = (pos[16] - lowest)/50
            if (p > min_time) and (p < max_time):
                #print('keep: ', track_id)
                pos[16] = p
                track.append(np.array(pos))
            else:
                pass
                #print('lose: ', track_id)
                
        if track != []:
            split_track.append(np.array(track))
            split_target.append(targets[track_id])
            split_trial.append(trials[track_id])

        track_id += 1

    print('COMPLETED FILTER: ', len(split_track))

    return np.array(split_track, dtype=object), np.array(split_target), np.array(split_trial)