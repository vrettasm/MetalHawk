# INFO:
__program__ = "MetalHawk data extractor"
__author__ = 'Gianmattia Sgueglia'
__email__ = 'gianmattia.sgueglia@unina.it'

import __main__

__main__.pymol_argv = ['pymol', '-qcv']

import pymol
from pymol import cmd, stored

pymol.finish_launching()

import os
import argparse
import numpy as np
from scipy.spatial.distance import pdist

parser = argparse.ArgumentParser()
parser.add_argument('--inputdir', type=str, required=True,
                    help='Path to a directory containing .pdb or .cif files from which to extract metal sites')
parser.add_argument('--outputdir', type=str, required=True,
                    help='Name of the directory where metal sites will be saved as .pdb files')
parser.add_argument('--build_crystal', type=bool, default=True,
                    help='Whether or not to build the crystal around the original structure')
parser.add_argument('--remove_alter', type=bool, default=True,
                    help='Whether or not to remove alternative conformations before extraction')
parser.add_argument('--radius', type=float, default=10.0,
                    help='Radius of the sphere to be extracted centered on the metal atom')
args = parser.parse_args()

metlist_pdb = ['TI', 'V', 'CR', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN',
               'ZR', 'NB', 'MO', 'TC', 'RU', 'RH', 'PD', 'AG', 'CD',
               'HF', 'TA', 'W', 'RE', 'OS', 'IR', 'PT', 'AU', 'HG']
metlist_csd = ['Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
               'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
               'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg']


def create_directory(directory_name):
    current_directory = os.getcwd()
    new_directory = os.path.join(current_directory, directory_name)
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
        print(f"Metal sites will be saved in '{directory_name}'.")
    else:
        print(f"Directory '{directory_name}' already exists.")


directory_name = args.outputdir
create_directory(directory_name)


def check_duplicates(selection_name):
    primary_coords = cmd.get_coords('all within 4 of %' + selection_name)
    distances = pdist(primary_coords, 'euclidean')
    # print(np.amin(distances))
    if np.amin(distances) < 0.005:
        duplicates_flag = 1
        print(f"Duplicated atoms found! Cleaning up metal site.")
    else:
        duplicates_flag = 0
    return duplicates_flag


def remove_duplicates():
    sph_model = cmd.get_model("all within 6 of %metal")
    for satom in sph_model.atom:
        cmd.select("sele_atom", "s. " + str(satom.segi) + " & c. " + str(satom.chain) +
                   " & i. " + str(satom.resi) + " & r. " + str(satom.resn) + " & n. " + str(satom.name))
        cmd.remove("all near_to 0.1 of %sele_atom")
    return


def get_metal_spheres():
    print(f"Looking for transition metals")
    cmd.remove("e. H")
    if args.remove_alter:
        cmd.remove("not alt ''+A")
        cmd.alter("all", "alt=''")
    model = cmd.get_model('structure')
    index = 0
    for atom in model.atom:
        if atom.symbol in metlist_pdb or atom.symbol in metlist_csd:
            print(f"Found '{atom.symbol}' atom, extracting site....")
            metal_coords = atom.coord

            print(f"Metal coordinates are '{metal_coords}'")
            xm = metal_coords[0]
            ym = metal_coords[1]
            zm = metal_coords[2]
            cmd.select('metal', '((x>' + str(xm - 0.05) + ' and x<' + str(xm + 0.05) + ') and (y>' + str(ym - 0.05) +
                       ' and y<' + str(ym + 0.05) + ') and (z>' + str(zm - 0.05) + ' and z<' + str(zm + 0.05) + '))')
            # print(cmd.count_atoms('metal'))
            duplicates_flag = check_duplicates('metal')
            cmd.select('sphr', 'all within ' + str(args.radius) + ' of %metal')
            cmd.create('sphere', 'sphr')
            if duplicates_flag:
                remove_duplicates()
            # print(cmd.count_atoms('metal'))
            if cmd.count_atoms('sphere') > 2000:
                print(f"============> Warning: Sphere containing a lot of atoms detected."
                      " It might be a problem with symmetry operations.")
            cmd.alter_state(1, 'sphere', "x=float(x)-" + str(xm))
            cmd.alter_state(1, 'sphere', "y=float(y)-" + str(ym))
            cmd.alter_state(1, 'sphere', "z=float(z)-" + str(zm))
            cmd.sort("all")
            cmd.save(os.path.join(args.outputdir, fileroot + "_" + str(index) + ".pdb"), 'sphere', 1, "pdb")
            cmd.remove('sphere')
            index += 1
            print(f"Done!")
    return


for filename in os.listdir(args.inputdir):
    if filename.endswith('.cif') or filename.endswith('.pdb'):
        cmd.reinitialize()
        cmd.load(os.path.join(args.inputdir, filename))
        fileroot = filename.split('.')[0]
        print(f"----------------------------| Loaded '{filename}', now processing... |----------------------------")
        objname = str(cmd.get_object_list("all")[0])
        cmd.set_name(objname, 'structure')
        alternative_locations = cmd.count_atoms("not alt ''+A")
        if alternative_locations > 0 and not args.remove_alter:
            print(f"============> Warning: Alternative conformations present. This might cause issues.")

        if args.build_crystal:
            cmd.symexp('crystal', 'structure', 'all', 10)
            objectlist = cmd.get_object_list("all")

            for objidx in range(0, len(objectlist), 1):
                objectc = objectlist[objidx]
                cmd.alter(objectc, "segi=" + str(objidx))
                cmd.sort()
        get_metal_spheres()
