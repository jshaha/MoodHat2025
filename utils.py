import random
import string

def generate_random_string(length=10):
    # Define the character set (alphanumeric in this case)
    charset = string.ascii_letters + string.digits
    # Generate a random string of the specified length using the random.choice method
    random_string = ''.join(random.choice(charset) for _ in range(length))
    return random_string

nanVal = float('nan')


















