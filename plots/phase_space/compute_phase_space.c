#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <hdf5.h>

#define NTYPES 6
#define IDX(i, j) i*3+j

struct Part
{
    long long ID;
    double Pos[3];
    double Vel[3];
    double Acc[3];
    long index;
};

void compute_Nchunk(int *Nchunk_id, int *Nchunk_snap){
    // TODO: compute these according to memory requirements
    *Nchunk_id = 64;
    *Nchunk_snap = 12;
    return;
}

int compute_nsnap(char* basepath){
    char output_dir[1000], fname[1000];
    sprintf(output_dir, "%s/output/", basepath);

    int i=0;
    sprintf(fname, "%s/snapdir_%03d/snapshot_%03d.0.hdf5", output_dir, i, i);
    while (access(fname, F_OK) == 0)
    {
        i++;
        sprintf(fname, "%s/snapdir_%03d/snapshot_%03d.0.hdf5", output_dir, i, i);
    }
    
    return i;
}

void read_header_attribute(hid_t file_id, hid_t DTYPE, char* attr_name, void *buf)
{
    hid_t header = H5Gopen(file_id, "/Header", H5P_DEFAULT);
    hid_t hdf5_attribute = H5Aopen(header, attr_name, H5P_DEFAULT);
    H5Aread(hdf5_attribute, DTYPE, buf);
    H5Aclose(hdf5_attribute);
    H5Gclose(header);
    return;
}

void read_parttype_ids(char *output_dir, int snap_idx, int PartType, long long **output_buf)
{
    char grp_name[100], dset_name[100], fname[1000];
    // read number of files and total number of particles
    int Nfiles;
    uint NumPart_Total[NTYPES], NumPart_ThisFile[NTYPES];
    long long *IDs_ThisFile;

    // read header attributes from first snapshot
    sprintf(fname, "%s/snapdir_%03d/snapshot_%03d.%d.hdf5", output_dir, snap_idx, snap_idx, 0);
    hid_t file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    read_header_attribute(file_id, H5T_NATIVE_INT, "NumFilesPerSnapshot", &Nfiles);
    read_header_attribute(file_id, H5T_NATIVE_UINT, "NumPart_Total", NumPart_Total);

    // allocate output buffer
    *output_buf = (long long *) malloc(sizeof(long long) * NumPart_Total[PartType]);

    sprintf(grp_name, "/PartType%d", PartType);
    sprintf(dset_name, "/PartType%d/%s", PartType, "ParticleIDs");

    H5Fclose(file_id);

    printf("fname=%s, Nfiles=%d\n", fname, Nfiles);

    // printf("grp_name=%s, dset_name=%s\n", grp_name, dset_name);

    // loop through files and copy data into output
    int NumPartCum= 0;
    for(int i=0; i<Nfiles; i++){
        sprintf(fname, "%s/snapdir_%03d/snapshot_%03d.%d.hdf5", output_dir, snap_idx, snap_idx, i);
        // printf("NumPartCum=%d, reading %s\n", NumPartCum, fname);
        hid_t file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

        read_header_attribute(file_id, H5T_NATIVE_UINT, "NumPart_ThisFile", NumPart_ThisFile);
        IDs_ThisFile = (long long *) malloc(sizeof(long long) * NumPart_ThisFile[PartType]);

        hid_t grp = H5Gopen(file_id, grp_name, H5P_DEFAULT);
        hid_t dset = H5Dopen(grp, dset_name, H5P_DEFAULT);
        H5Dread(dset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, IDs_ThisFile);

        // copy from this file into output buffer
        memcpy(&((*output_buf)[NumPartCum]), IDs_ThisFile, sizeof(long long) * NumPart_ThisFile[PartType]);
        // printf("NumPart_ThisFile=%d\n", NumPart_ThisFile[PartType]);
        NumPartCum += NumPart_ThisFile[PartType];

        H5Dclose(dset);
        H5Gclose(grp);
        H5Fclose(file_id);
    }

    return;
}

void read_parttype_vec(char *output_dir, int snap_idx, int PartType, char *property, double **output_buf)
{
    char grp_name[100], dset_name[100], fname[1000];
    // read number of files and total number of particles
    int Nfiles;
    uint NumPart_Total[NTYPES], NumPart_ThisFile[NTYPES];
    double *Vec_ThisFile;

    // read header attributes from first snapshot
    sprintf(fname, "%s/snapdir_%03d/snapshot_%03d.%d.hdf5", output_dir, snap_idx, snap_idx, 0);
    hid_t file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    read_header_attribute(file_id, H5T_NATIVE_INT, "NumFilesPerSnapshot", &Nfiles);
    read_header_attribute(file_id, H5T_NATIVE_UINT, "NumPart_Total", NumPart_Total);

    // allocate output buffer
    *output_buf = (double *) malloc(3 * sizeof(double) * NumPart_Total[PartType]);

    sprintf(grp_name, "/PartType%d", PartType);
    sprintf(dset_name, "/PartType%d/%s", PartType, property);

    H5Fclose(file_id);

    printf("fname=%s, Nfiles=%d\n", fname, Nfiles);

    // printf("grp_name=%s, dset_name=%s\n", grp_name, dset_name);

    // loop through files and copy data into output
    int NumPartCum= 0;
    for(int i=0; i<Nfiles; i++){
        sprintf(fname, "%s/snapdir_%03d/snapshot_%03d.%d.hdf5", output_dir, snap_idx, snap_idx, i);
        // printf("NumPartCum=%d, reading %s\n", NumPartCum, fname);
        hid_t file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

        read_header_attribute(file_id, H5T_NATIVE_UINT, "NumPart_ThisFile", NumPart_ThisFile);
        Vec_ThisFile = (double *) malloc(3 * sizeof(double) * NumPart_ThisFile[PartType]);

        hid_t grp = H5Gopen(file_id, grp_name, H5P_DEFAULT);
        hid_t dset = H5Dopen(grp, dset_name, H5P_DEFAULT);
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, Vec_ThisFile);

        // copy from this file into output buffer
        memcpy(&((*output_buf)[3 * NumPartCum]), Vec_ThisFile, 3 * sizeof(double) * NumPart_ThisFile[PartType]);
        // printf("NumPart_ThisFile=%d\n", NumPart_ThisFile[PartType]);
        NumPartCum += NumPart_ThisFile[PartType];

        H5Dclose(dset);
        H5Gclose(grp);
        H5Fclose(file_id);
    }

    return;
}

    // double *DiskCoordinates;
    // DiskCoordinates = (double *)malloc(sizeof(double) * NumPart_Total[2] * 3);
    // read_parttype_dataset(file_id, H5T_NATIVE_DOUBLE, 2, "Coordinates", DiskCoordinates);
    // printf("DiskCoordinates[0]=%g|%g|%g  [1]=%g|%g|%g\n", DiskCoordinates[IDX(0, 0)], DiskCoordinates[IDX(0, 1)], DiskCoordinates[IDX(0, 2)],
    //                                                       DiskCoordinates[IDX(1, 0)], DiskCoordinates[IDX(1, 1)], DiskCoordinates[IDX(1, 2)]);

long long cmprID (const void *a, const void *b){
    struct Part *partA = (struct Part *)a;
    struct Part *partB = (struct Part *)b;
    return (partA->ID - partB->ID);
}

void get_part(char* basepath, int Nsnap, int PartType, struct Part ** part)
{
    // read final snapshot
    char fname[1000], output_dir[1000];
    uint NumPart_Total[NTYPES];
    sprintf(fname, "%s/output/snapdir_%03d/snapshot_%03d.0.hdf5", basepath, Nsnap-1, Nsnap-1);
    printf("fname=%s\n", fname);
    hid_t file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

    // read num part total
    read_header_attribute(file_id, H5T_NATIVE_UINT, "NumPart_Total", NumPart_Total);
    
    // read disk ids
    long long *DiskIDs;
    sprintf(output_dir, "%s/output/", basepath);
    read_parttype_ids(output_dir, Nsnap-1, PartType, &DiskIDs);
    
    // read phase space coordinates
    double *DiskPos, *DiskVel, *DiskAcc;
    read_parttype_vec(output_dir, Nsnap-1, PartType, "Coordinates", &DiskPos);
    read_parttype_vec(output_dir, Nsnap-1, PartType, "Velocities", &DiskVel);
    read_parttype_vec(output_dir, Nsnap-1, PartType, "Acceleration", &DiskAcc);

    *part = (struct Part *)malloc(sizeof(struct Part) * NumPart_Total[PartType]);

    printf("DiskPos[0] = %g|%g|%g\n", DiskPos[IDX(0, 0)], DiskPos[IDX(0, 1)], DiskPos[IDX(0, 2)]);

    // load from separate arrays into structured array
    int i, j;
    for(i=0; i<NumPart_Total[PartType]; i++){
        // part[i]->ID = DiskIDs[i];
        (*part)[i].ID = DiskIDs[i];
        (*part)[i].index = i;
        for(j=0; j<3; j++){
            (*part)[i].Pos[j] = DiskPos[IDX(i, j)];
            (*part)[i].Vel[j] = DiskVel[IDX(i, j)];
            (*part)[i].Acc[j] = DiskAcc[IDX(i, j)];
        }
    }

    // now 

    // printf("DiskIDs[0]=%lld, DiskIDs[100]=%lld, DiskIDs[last]=%lld\n", DiskIDs[0], DiskIDs[100], DiskIDs[NumPart_Total[2]-1]);

    herr_t status = H5Fclose(file_id);

}

long long * get_IDs_from_part(struct Part * part, uint Npart){
    long long *IDs;
    IDs = (long long *)malloc(Npart * sizeof(long long));

    for(int i=0; i<Npart; i++){
        IDs[i] = part[i].ID;
    }
    return IDs;
}

    // sort disk ids
    // qsort(DiskIDs, NumPart_Total[2], sizeof(long long), cmprllong);

int main(int argc, char* argv[]) {
    char name[100];
    char lvl[100];
    int Nchunk_id, Nchunk_snap, Nsnap;
    char basepath[1000], output_dir[1000], fname[1000];
    int *id_chunks_disk, *indices_chunks;
    uint NumPart_Total[NTYPES];

    // Check to make sure right number of arguments.
    if(argc != 3){
        printf("Usage: ./compute_phase_space.o name lvl\n");
        return -1;
    }

    // Copy name and lvl, print.
    strcpy(name, argv[1]);
    strcpy(lvl, argv[2]);

    printf("Running for name=%s, lvl=%s\n", name, lvl);

    // compute Nchunks
    compute_Nchunk(&Nchunk_id, &Nchunk_snap);
    printf("Nchunk_id=%d, Nchunk_snap=%d\n", Nchunk_id, Nchunk_snap);

    // Write down basepath/output dir and search for the number of snapshots
    sprintf(basepath, "../../runs/%s/%s/", name, lvl);
    sprintf(output_dir, "%s/output/", basepath);
    Nsnap = compute_nsnap(basepath);

    sprintf(fname, "%s/snapdir_%03d/snapshot_%03d.0.hdf5", output_dir, 0, 0);
    hid_t file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    read_header_attribute(file_id, H5T_NATIVE_UINT, "NumPart_Total", NumPart_Total);
    H5Fclose(file_id);

    // Pull out particles from the last snapshot
    struct Part *DiskPart, *BulgePart;
    get_part(basepath, Nsnap, 2, &DiskPart);
    get_part(basepath, Nsnap, 3, &BulgePart);

    qsort(DiskPart, NumPart_Total[2], sizeof(struct Part), cmprID);
    qsort(BulgePart, NumPart_Total[3], sizeof(struct Part), cmprID);

    // construct sorted ID list
    long long *DiskIDs, *BulgeIDs;
    DiskIDs = get_IDs_from_part(DiskPart, NumPart_Total[2]);
    BulgeIDs = get_IDs_from_part(BulgePart, NumPart_Total[3]);
    


    int i;
    i = 0;

    printf("DiskIDs[0]=%lld, DiskIDs[100]=%lld, DiskIDs[last]=%lld\n", DiskPart[0].ID, DiskPart[100].ID, DiskPart[NumPart_Total[2]-1].ID);
    printf("BulgeIDs[0]=%lld, BulgeIDs[100]=%lld, BulgeIDs[last]=%lld\n", BulgePart[0].ID, BulgePart[100].ID, BulgePart[NumPart_Total[3]-1].ID);

    printf("DiskIDs[0]=%lld, DiskIDs[100]=%lld, DiskIDs[last]=%lld\n", DiskIDs[0], DiskIDs[100], DiskIDs[NumPart_Total[2]-1]);
    printf("BulgeIDs[0]=%lld, BulgeIDs[100]=%lld, BulgeIDs[last]=%lld\n", BulgeIDs[0], BulgeIDs[100], BulgeIDs[NumPart_Total[3]-1]);

    printf("part[0] ID=%lld, Pos=%g|%g|%g, Vel=%g|%g|%g, Acc=%g|%g|%g\n", BulgePart[i].ID, BulgePart[i].Pos[0], BulgePart[i].Pos[1], BulgePart[i].Pos[2],
                                                                          BulgePart[i].Vel[0], BulgePart[i].Vel[1], BulgePart[i].Vel[2], 
                                                                          BulgePart[i].Acc[0], BulgePart[i].Acc[1], BulgePart[i].Acc[2]);

    // Compute nsnap


    return 0;
}
