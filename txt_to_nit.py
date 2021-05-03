input_file_name = 'results.txt'
output_file_name = 'results_converted.nit'

def process_table(text_file, out):
    line = text_file.readline()
    while line != '\n' and line != '':
        tokens = line.split()
        out.append(list(map(float, tokens)))
        line = text_file.readline()

wl_400 = []
wl_500 = []
wl_600 = []
wl_700 = []

with open(input_file_name) as text_file:
    line = text_file.readline()
    while line != '':
        tokens = line.split()
        if len(tokens) == 0:
            line = text_file.readline()
            continue
        if tokens[0] == 'wavelength' and tokens[1] == '400':
            process_table(text_file, wl_400)
        if tokens[0] == 'wavelength' and tokens[1] == '500':
            process_table(text_file, wl_500)
        if tokens[0] == 'wavelength' and tokens[1] == '600':
            process_table(text_file, b)
        if tokens[0] == 'wavelength' and tokens[1] == '700':
            process_table(text_file, b)
        line = text_file.readline()

data.append(wl_400)
data.append(wl_500)
data.append(wl_600)
data.append(wl_700)

pp = PostProcessor(PPDataUnits.RADIANCE, [400,500,600,700], *data)

pp.SaveToHDR(output_file_name, overwrite = OverwriteMode.OVERWRITE)