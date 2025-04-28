import numpy as np 
from IPython.display import display 


def pretty_print(df):
    """ Only convert numbers to 2 decimal places with a comma separator """
    return display(df.map(lambda x: "{:,.2f}".format(x) if isinstance(x, (int, float)) else x))
    

def human_format(num):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        # add more suffixes if you need them
        # if .0 is the decimal part, remove it
        if num == int(num):
            return '%d%s' % (np.abs(num), ['', 'K', 'M', 'B'][magnitude])
        else:
             
            return '%.1f%s' % (np.abs(num), ['', 'K', 'M', 'B'][magnitude])


if __name__ == "__main__":
    # Example usage
    print(human_format(1234567))  # Output: '1.2M'
    print(human_format(-9876543210))  # Output: '-9.9B'
    print(human_format(100000))