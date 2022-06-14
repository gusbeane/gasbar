#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <hdf5.h>

#define NTYPES 6
#define IDX(i, j) i*3+j

int Nsnap;
uint NumPart_Total[NTYPES];
double MassTable[NTYPES];
int Nfiles;

struct Part{
    double Pos[3];
    double Mass;
};

void read_header_attribute(hid_t file_id, hid_t DTYPE, char* attr_name, void *buf){
    hid_t header = H5Gopen(file_id, "/Header", H5P_DEFAULT);
    hid_t hdf5_attribute = H5Aopen(header, attr_name, H5P_DEFAULT);
    H5Aread(hdf5_attribute, DTYPE, buf);
    H5Aclose(hdf5_attribute);
    H5Gclose(header);
    return;
}

void read_parttype(char *output_dir, int snapnum, int ptype, struct Part** output_part){
    char fname[1000], grp_name[1000];
    hid_t file_id, grp_id, dset;
    int i, k;
    long long Ncum, j;
    uint NumPart_ThisFile[NTYPES];
    double *Pos, *Mass;

    *output_part = (struct Part*) malloc(sizeof(struct Part) * NumPart_Total[ptype]);

    Ncum = 0;
    Nfiles = 1;
    for(i=0; i<Nfiles; i++){
        sprintf(fname, "%s/snapdir_%03d/snapshot_%03d.%d.hdf5", output_dir, snapnum, snapnum, i);
        file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
        read_header_attribute(file_id, H5T_NATIVE_UINT, "NumPart_ThisFile", NumPart_ThisFile);
        Pos = (double *)malloc(sizeof(double) * 3 * NumPart_ThisFile[ptype]);
        Mass = (double *)malloc(sizeof(double) * NumPart_ThisFile[ptype]);

        sprintf(grp_name, "PartType%d", ptype);
        printf("grp_name=%s\n", grp_name);
        grp_id = H5Gopen(file_id, grp_name, H5P_DEFAULT);

        // Read coordinates.
        dset = H5Dopen(grp_id, "Coordinates", H5P_DEFAULT);
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &Pos);
        H5Dclose(dset);

        // if(MassTable[ptype] == 0.0){
        //     dset = H5Dopen(grp_id, "Masses", H5P_DEFAULT);
        //     H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &Mass);
        //     H5Dclose(dset);
        // }

        H5Gclose(grp_id);
        H5Fclose(file_id);

        for(j=0; j<NumPart_ThisFile[ptype]; j++){
            for(k=0; k<3; k++){
                // printf("IDX(j-Ncum, k)=%lld\n", IDX(j-Ncum, k));
                (*output_part)[j+Ncum].Pos[k] = Pos[IDX(j, k)];
                if(MassTable[ptype]==0.0)
                    (*output_part)[j+Ncum].Mass = Mass[j];
                else
                    (*output_part)[j+Ncum].Mass = MassTable[ptype];
            }
        }
        Ncum += NumPart_ThisFile[ptype];

        free(Pos);
        free(Mass);
    } // ends loop over files

    return;
}

void compute_fourier_component(char *output_dir, int snapnum){
    char fname[1000];
    struct Part *DiskP, *BulgeP, *StarP;

    sprintf(fname, "%s/snapdir_%03d/snapshot_%03d.%d.hdf5", output_dir, snapnum, snapnum, 0);


    read_parttype(output_dir, snapnum, 2, &DiskP);
    read_parttype(output_dir, snapnum, 3, &BulgeP);
    
    for(int i=0; i<10; i++)
        printf("DiskP.Pos = %g|%g|%g\n", DiskP[i].Pos[0], DiskP[i].Pos[1], DiskP[i].Pos[2]);



}

void compute_Nsnap(char* output_dir){
    char fname[1000];
    int i=0;

    sprintf(fname, "%s/snapdir_%03d/snapshot_%03d.0.hdf5", output_dir, i, i);
    while (access(fname, F_OK) == 0){
        i++;
        sprintf(fname, "%s/snapdir_%03d/snapshot_%03d.0.hdf5", output_dir, i, i);
    }
    Nsnap = i;
    Nsnap = 1;
}

void preprocess_header(char *output_dir, int snapnum){
    char fname[2000];
    hid_t file_id;

    sprintf(fname, "%s/snapdir_%03d/snapshot_%03d.%d.hdf5", output_dir, snapnum, snapnum, 0);
    file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    read_header_attribute(file_id, H5T_NATIVE_INT, "NumFilesPerSnapshot", &Nfiles);
    read_header_attribute(file_id, H5T_NATIVE_UINT, "NumPart_Total", NumPart_Total);
    read_header_attribute(file_id, H5T_NATIVE_DOUBLE, "MassTable", MassTable);
    H5Fclose(file_id);

}

int main(int argc, char*argv[]){
    char name[100], lvl[100], basepath[500], output_dir[1000];

    if(argc != 3){
        // if(rank == 0)
            printf("Usage: ./compute_phase_space.o name lvl\n");
        exit(1);
    }

    printf("0\n");

    // read name and lvl from command line
    strcpy(name, argv[1]);
    strcpy(lvl, argv[2]);

    // construct basepath and output directory
    sprintf(basepath, "../../runs/%s/%s/", name, lvl);
    sprintf(output_dir, "%s/output/", basepath);

    printf("1\n");


    compute_Nsnap(output_dir);
    for(int i=0; i<Nsnap; i++){
        preprocess_header(output_dir, i);
        compute_fourier_component(output_dir, i);
        printf("i=%d\n", i);
    }

    printf("2\n");

    
    printf("Hello, world, Nsnap=%d\n", Nsnap);

    

    return 0;
}
