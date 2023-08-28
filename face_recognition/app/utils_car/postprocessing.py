import re

char_to_int_dict = {'O': '0',
                    "D" : "0",
                    'I': '1',
                    'A': '4',
                    'G': '6',
                    'S': '5',
                    "B": "8",
                    "Z": "2"}

int_to_char_dict = {'0': 'O',
                    '1': 'I',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S',
                    "8": "B",
                    "2": "Z"}

def correct_old_plate(text):

    text = list(text)
    
    text[0] = int_to_char_dict[text[0]] if text[0] in int_to_char_dict.keys() else text[0]
    text[1] = int_to_char_dict[text[1]] if text[1] in int_to_char_dict.keys() else text[1]
    text[2] = int_to_char_dict[text[2]] if text[2] in int_to_char_dict.keys() else text[2]

    text[3] = char_to_int_dict[text[3]] if text[3] in char_to_int_dict.keys() else text[3]
    text[4] = char_to_int_dict[text[4]] if text[4] in char_to_int_dict.keys() else text[4]
    text[5] = char_to_int_dict[text[5]] if text[5] in char_to_int_dict.keys() else text[5]
    text[6] = char_to_int_dict[text[6]] if text[6] in char_to_int_dict.keys() else text[6]

    text = "".join(text)
    
    return text

def validate_old_plate(text):

    pattern_old_plate = "^[A-Z]{3}[0-9]{4}"
    regex_old_plate = re.compile(pattern_old_plate)
    text_len = len(text)

    if text_len == 8:
        alternative_left = correct_old_plate(text[:7])
        alternative_right = correct_old_plate(text[1:])
        if re.fullmatch(regex_old_plate, alternative_left):
            return True, alternative_left
        if re.fullmatch(regex_old_plate, alternative_right):
            return True, alternative_right   
        return False, "Leitura incorreta"
    
    if text_len == 7:
        text = correct_old_plate(text)
        if re.fullmatch(regex_old_plate, text):
            return True, text
        else:
            return False, "Leitura incorreta"

def correct_new_plate(text):
    
    text = list(text)
    
    text[0] = int_to_char_dict[text[0]] if text[0] in int_to_char_dict.keys() else text[0]
    text[1] = int_to_char_dict[text[1]] if text[1] in int_to_char_dict.keys() else text[1]
    text[2] = int_to_char_dict[text[2]] if text[2] in int_to_char_dict.keys() else text[2]

    text[3] = char_to_int_dict[text[3]] if text[3] in char_to_int_dict.keys() else text[3]
    
    text[4] = int_to_char_dict[text[4]] if text[4] in int_to_char_dict.keys() else text[4]
    
    text[5] = char_to_int_dict[text[5]] if text[5] in char_to_int_dict.keys() else text[5]
    text[6] = char_to_int_dict[text[6]] if text[6] in char_to_int_dict.keys() else text[6]
    
    text = "".join(text)
    
    return text

def validate_new_plate(text):
    
    pattern_new_plate = "^[A-Z]{3}[0-9][A-Z][0-9]{2}"
    regex_new_plate = re.compile(pattern_new_plate)
    text_len = len(text)
    

    if text_len == 8:
        alternative_left = correct_new_plate(text[:7])
        alternative_right = correct_new_plate(text[1:])
        if re.fullmatch(regex_new_plate, alternative_left):
            return True, alternative_left
        if re.fullmatch(regex_new_plate, alternative_right):
            return True, alternative_right   
        return False, "Leitura incorreta"
    
    if text_len == 7:
        text = correct_new_plate(text)
        if re.fullmatch(regex_new_plate, text):
            return True, text
        return False, "Leitura incorreta"

def validate_regex(text):
    pattern_new_plate = "^[A-Z]{3}[0-9][A-Z][0-9]{2}"
    regex_new_plate = re.compile(pattern_new_plate)

    pattern_old_plate = "^[A-Z]{3}[0-9]{4}"
    regex_old_plate = re.compile(pattern_old_plate)
    
    text_len = len(text)

    if text_len == 8:
        alternative_left = text[:7]
        alternative_right = text[1:]
        if re.fullmatch(regex_new_plate, alternative_left) or re.fullmatch(regex_old_plate, alternative_left):
            return True, alternative_left
        if re.fullmatch(regex_new_plate, alternative_right) or re.fullmatch(regex_old_plate, alternative_right):
            return True, alternative_right   
        return False, "Leitura incorreta"
    
    if text_len == 7:
        if re.fullmatch(regex_new_plate, text) or re.fullmatch(regex_old_plate, text):
            return True, text
        return False, "Leitura incorreta"

def validate_plate(text):
    text = text.replace("-","")
    text = text.replace(".","")
    text = text.upper()
    if len(text)< 7 or len(text) >8:
        return False, text[:7]
    is_regex_plate, plate = validate_regex(text)
    if is_regex_plate:
        return True, plate
    
    is_old_plate, old_plate = validate_old_plate(text)
    if is_old_plate:
        return True, old_plate
        
    is_new_plate, new_plate = validate_new_plate(text)
    if is_new_plate:
        return True, new_plate
        
    
    
    return False, text[:7]