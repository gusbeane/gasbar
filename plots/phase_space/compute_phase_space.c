#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <hdf5.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>
#include <mpi.h>

#define NTYPES 6
#define IDX(i, j) i*3+j

uint NumPart_Total_LastSnap[NTYPES];

struct Part
{
    long long ID;
    double Pos[3];
    double Vel[3];
    double Acc[3];
    long index;
};

hid_t my_H5Gopen(hid_t loc_id, const char *groupname, hid_t fapl_id)
{
  hid_t group = H5Gopen(loc_id, groupname, fapl_id);

  if(group < 0)
    {
      printf("Error detected in HDF5: unable to open group %s\n", groupname);
      exit(1);
    }
  return group;
}

void compute_Nchunk(int *Nchunk_id, int *Nchunk_snap){
    // TODO: compute these according to memory requirements
    *Nchunk_id = 64;
    *Nchunk_snap = 128;
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
    hid_t header = my_H5Gopen(file_id, "/Header", H5P_DEFAULT);
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

    // printf("grp_name=%s, dset_name=%s\n", grp_name, dset_name);

    // loop through files and copy data into output
    int NumPartCum= 0;
    for(int i=0; i<Nfiles; i++){
        sprintf(fname, "%s/snapdir_%03d/snapshot_%03d.%d.hdf5", output_dir, snap_idx, snap_idx, i);
        // printf("NumPartCum=%d, reading %s\n", NumPartCum, fname);
        hid_t file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

        read_header_attribute(file_id, H5T_NATIVE_UINT, "NumPart_ThisFile", NumPart_ThisFile);
        IDs_ThisFile = (long long *) malloc(sizeof(long long) * NumPart_ThisFile[PartType]);

        hid_t grp = my_H5Gopen(file_id, grp_name, H5P_DEFAULT);
        hid_t dset = H5Dopen(grp, dset_name, H5P_DEFAULT);
        H5Dread(dset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, IDs_ThisFile);

        // copy from this file into output buffer
        memcpy(&((*output_buf)[NumPartCum]), IDs_ThisFile, sizeof(long long) * NumPart_ThisFile[PartType]);
        // printf("NumPart_ThisFile=%d\n", NumPart_ThisFile[PartType]);
        NumPartCum += NumPart_ThisFile[PartType];

        H5Dclose(dset);
        H5Gclose(grp);
        H5Fclose(file_id);
        free(IDs_ThisFile);
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

    // printf("fname=%s, Nfiles=%d\n", fname, Nfiles);

    // printf("grp_name=%s, dset_name=%s\n", grp_name, dset_name);

    // loop through files and copy data into output
    int NumPartCum= 0;
    for(int i=0; i<Nfiles; i++){
        sprintf(fname, "%s/snapdir_%03d/snapshot_%03d.%d.hdf5", output_dir, snap_idx, snap_idx, i);
        // printf("NumPartCum=%d, reading %s\n", NumPartCum, fname);
        hid_t file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

        read_header_attribute(file_id, H5T_NATIVE_UINT, "NumPart_ThisFile", NumPart_ThisFile);
        Vec_ThisFile = (double *) malloc(3 * sizeof(double) * NumPart_ThisFile[PartType]);

        hid_t grp = my_H5Gopen(file_id, grp_name, H5P_DEFAULT);
        hid_t dset = H5Dopen(grp, dset_name, H5P_DEFAULT);
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, Vec_ThisFile);

        // copy from this file into output buffer
        memcpy(&((*output_buf)[3 * NumPartCum]), Vec_ThisFile, 3 * sizeof(double) * NumPart_ThisFile[PartType]);
        // printf("NumPart_ThisFile=%d\n", NumPart_ThisFile[PartType]);
        NumPartCum += NumPart_ThisFile[PartType];

        H5Dclose(dset);
        H5Gclose(grp);
        H5Fclose(file_id);
        free(Vec_ThisFile);
    }

    return;
}

int cmprID (const void *a, const void *b){
    struct Part *partA = (struct Part *)a;
    struct Part *partB = (struct Part *)b;
    if (partA->ID - partB->ID > 0)
        return 1;
    if (partA->ID - partB->ID < 0)
        return -1;
    return 0;
}

void get_part(char* output_dir, int SnapIdx, int PartType, struct Part ** part)
{
    // read final snapshot
    char fname[1000];
    uint NumPart_Total[NTYPES];
    sprintf(fname, "%s/snapdir_%03d/snapshot_%03d.0.hdf5", output_dir, SnapIdx, SnapIdx);
    // printf("fname=%s\n", fname);
    hid_t file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

    // read num part total
    read_header_attribute(file_id, H5T_NATIVE_UINT, "NumPart_Total", NumPart_Total);
    
    // read disk ids
    long long *DiskIDs;
    read_parttype_ids(output_dir, SnapIdx, PartType, &DiskIDs);
    
    // read phase space coordinates
    double *DiskPos, *DiskVel, *DiskAcc;
    read_parttype_vec(output_dir, SnapIdx, PartType, "Coordinates", &DiskPos);
    read_parttype_vec(output_dir, SnapIdx, PartType, "Velocities", &DiskVel);
    read_parttype_vec(output_dir, SnapIdx, PartType, "Acceleration", &DiskAcc);

    *part = (struct Part *)malloc(sizeof(struct Part) * NumPart_Total[PartType]);

    // printf("DiskPos[0] = %g|%g|%g\n", DiskPos[IDX(0, 0)], DiskPos[IDX(0, 1)], DiskPos[IDX(0, 2)]);

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

    free(DiskPos);
    free(DiskVel);
    free(DiskAcc);
    free(DiskIDs);

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

void array_split_int(int Nchunk, int *List, int NList, int ***OutList, int **OutNPerList){
    // splits the array List, of size Nlist, into Nchunk lists, stored in OutList, where each OutList
    // has size stored in OutNPerList

    // First, compute the number of items per output list.
    // printf("Nchunk=%d, NList=%d\n", Nchunk, NList);
    // printf("%g\n", ceil(NList/Nchunk));
    int NperListMax = (NList - 1)/Nchunk + 1;
    int NLeftOver = NperListMax * Nchunk - NList;

    // printf("NperListMax=%d\n", NperListMax);
    // printf("NLeftOver=%d\n", NLeftOver);

    // Allocate and write down the number of items in each chunk
    *OutNPerList = (int *)malloc(sizeof(int) * Nchunk);
    for(int i=0; i<Nchunk; i++){
        // printf("i=%d\n", i);
        (*OutNPerList)[i] = NperListMax;
        if(i >= (Nchunk - NLeftOver))
            (*OutNPerList)[i] -= 1;
    }

    // verify
    int chk = 0;
    for(int i=0; i<Nchunk; i++)
        chk += (*OutNPerList)[i];
    if(chk != NList)
        printf("WARNING chk=%d is not equal to NList=%d\n", chk, NList);

    // Now allocate output list for each chunk
    *OutList = (int **)malloc(sizeof(int *) * Nchunk);
    for(int i=0; i<Nchunk; i++){
        (*OutList)[i] = (int *)malloc(sizeof(int) * (*OutNPerList)[i]);
    }

    // Now copy the list into each chunk
    int Ncum = 0;
    for(int i=0; i<Nchunk; i++){
        memcpy((*OutList)[i], List + Ncum, sizeof(int) * (*OutNPerList)[i]);
        Ncum += (*OutNPerList)[i];
    }
}

void compute_chunk_start_end(int rank, int size, int Nchunk, int *ChunkStart, int *ChunkEnd){
    int NperMax = (Nchunk -1)/size + 1;
    int NLeftOver = NperMax * size - Nchunk;
    *ChunkStart = 0;
    int itr = 0;
    int i;
    for(i=0; i<rank; i++){
        itr = NperMax;
        if(i >= (size - NLeftOver))
            itr--;
        *ChunkStart += itr;
    }
    itr = NperMax;
    if(i >= (size - NLeftOver))
        itr--;
    *ChunkEnd = *ChunkStart + itr;
    return;
}

void array_split_llong(int Nchunk, long long *List, int NList, long long ***OutList, long long **OutNPerList){
    // splits the array List, of size Nlist, into Nchunk lists, stored in OutList, where each OutList
    // has size stored in OutNPerList

    // First, compute the number of items per output list.
    // printf("Nchunk=%d, NList=%d\n", Nchunk, NList);
    // printf("%g\n", ceil(NList/Nchunk));
    long long NperListMax = (NList - 1)/Nchunk + 1;
    long long NLeftOver = NperListMax * Nchunk - NList;

    // Allocate and write down the number of items in each chunk
    // printf("NperListMax=%lld\n", NperListMax);
    *OutNPerList = (long long *)malloc(sizeof(long long) * Nchunk);
    for(int i=0; i<Nchunk; i++){
        (*OutNPerList)[i] = NperListMax;
        if(i >= (Nchunk - NLeftOver))
            (*OutNPerList)[i] -= 1;
    }

    // verify
    long long chk = 0;
    for(int i=0; i<Nchunk; i++)
        chk += (*OutNPerList)[i];
    // printf("chk=%lld, NList=%lld\n", chk, NList);
    if(chk != NList)
        printf("WARNING chk=%lld is not equal to NList=%lld\n", chk, NList);

    // Now allocate output list for each chunk
    *OutList = (long long **)malloc(sizeof(long long *) * Nchunk);
    for(int i=0; i<Nchunk; i++){
        (*OutList)[i] = (long long *)malloc(sizeof(long long) * (*OutNPerList)[i]);
    }

    // Now copy the list into each chunk
    long long Ncum = 0;
    for(int i=0; i<Nchunk; i++){
        memcpy((*OutList)[i], List + Ncum, sizeof(long long) * (*OutNPerList)[i]);
        Ncum += (*OutNPerList)[i];
    }
}

void sort_by_id(long long *chunk_ids, long long Nids_in_chunk, struct Part * part, double **out_pos, double **out_vel, double **out_acc)
{
    long long itot=0;
    long long ichunk, chk_id;
    int j;

    for(ichunk=0; ichunk<Nids_in_chunk; ichunk++){
        // printf("chk_id=%lld, part[itot].ID=%lld\n", chk_id, part[itot].ID);
        chk_id = chunk_ids[ichunk];
        while(chk_id > part[itot].ID)
        {
            // printf("itot=%d\n", itot);
            itot++;
        }

        if(chk_id == part[itot].ID){
            for(j=0; j<3; j++){
                (*out_pos)[IDX(ichunk, j)] = part[itot].Pos[j];
                (*out_vel)[IDX(ichunk, j)] = part[itot].Vel[j];
                (*out_acc)[IDX(ichunk, j)] = part[itot].Acc[j];
            }
        }
        else{
            for(j=0; j<3; j++){
                (*out_pos)[IDX(ichunk, j)] = NAN;
                (*out_vel)[IDX(ichunk, j)] = NAN;
                (*out_acc)[IDX(ichunk, j)] = NAN;
            }
        }
    }
}

void process_snap_chunk(int i, char *output_dir, char *name, char *lvl, int *SnapChunk, int NSnapInChunk, int Nchunk_id,
                           long long **DiskIDsChunkList, long long *DiskIDsChunkListNumPer,
                           long long **BulgeIDsChunkList, long long *BulgeIDsChunkListNumPer){
    // i is the snap chunk idx
    // j loops through the disk id and bulge id chunks
    // we output into data/phase_space_name/tmp"i"/tmp"j".hdf5
    int j;
    char prefix[1000], fname[1000];
    uint NumPart_Total[NTYPES];
    struct stat st = {0};
    sprintf(prefix, "data/%s-%s/tmp%d", name, lvl, i);
    if (stat(prefix, &st) == -1) {
        mkdir(prefix, 0700);
    }

    double *Time;
    Time = (double *)malloc(sizeof(double) * NSnapInChunk);

    double **DiskPos, **DiskVel, **DiskAcc;
    double **BulgePos, **BulgeVel, **BulgeAcc;
    DiskPos = (double **)malloc(sizeof(double *) * Nchunk_id);
    DiskVel = (double **)malloc(sizeof(double *) * Nchunk_id);
    DiskAcc = (double **)malloc(sizeof(double *) * Nchunk_id);
    BulgePos = (double **)malloc(sizeof(double *) * Nchunk_id);
    BulgeVel = (double **)malloc(sizeof(double *) * Nchunk_id);
    BulgeAcc = (double **)malloc(sizeof(double *) * Nchunk_id);
    for(int ii=0; ii<Nchunk_id; ii++){
        DiskPos[ii] = (double *)malloc(sizeof(double) * NSnapInChunk * 3 * DiskIDsChunkListNumPer[ii]);
        DiskVel[ii] = (double *)malloc(sizeof(double) * NSnapInChunk * 3 * DiskIDsChunkListNumPer[ii]);
        DiskAcc[ii] = (double *)malloc(sizeof(double) * NSnapInChunk * 3 * DiskIDsChunkListNumPer[ii]);
        BulgePos[ii] = (double *)malloc(sizeof(double) * NSnapInChunk * 3 * BulgeIDsChunkListNumPer[ii]);
        BulgeVel[ii] = (double *)malloc(sizeof(double) * NSnapInChunk * 3 * BulgeIDsChunkListNumPer[ii]);
        BulgeAcc[ii] = (double *)malloc(sizeof(double) * NSnapInChunk * 3 * BulgeIDsChunkListNumPer[ii]);
    }

    // load in the snapshots
    struct Part *DiskPart, *BulgePart;
    double this_time;
    hid_t file_id;
    long long offset_disk[Nchunk_id], offset_bulge[Nchunk_id];
    for(j=0; j<Nchunk_id; j++){
        offset_disk[j] = 0;
        offset_bulge[j] = 0;
    }

    double *dpos_off, *dvel_off, *dacc_off;
    double *bpos_off, *bvel_off, *bacc_off;
    for(j=0; j<NSnapInChunk; j++){
        // read necessary header attributes
        sprintf(fname, "%s/snapdir_%03d/snapshot_%03d.0.hdf5", output_dir, SnapChunk[j], SnapChunk[j]);
        file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
        read_header_attribute(file_id, H5T_NATIVE_UINT, "NumPart_Total", NumPart_Total);
        read_header_attribute(file_id, H5T_NATIVE_DOUBLE, "Time", &this_time);
        H5Fclose(file_id);
        Time[j] = this_time;
        printf("processing snap %d (%d of %d), time=%g\n", SnapChunk[j], j, NSnapInChunk, Time[j]);

        // load in the snapshots
        get_part(output_dir, SnapChunk[j], 2, &DiskPart);
        get_part(output_dir, SnapChunk[j], 3, &BulgePart);

        // sort by ID
        qsort(DiskPart, NumPart_Total[2], sizeof(struct Part), cmprID);
        qsort(BulgePart, NumPart_Total[3], sizeof(struct Part), cmprID);

        // now loop through chunks and sort pos and vel into output
        for(int k=0; k<Nchunk_id; k++){
            dpos_off = &(DiskPos[k][offset_disk[k]]);
            dvel_off = &(DiskVel[k][offset_disk[k]]);
            dacc_off = &(DiskAcc[k][offset_disk[k]]);
            bpos_off = &(BulgePos[k][offset_bulge[k]]);
            bvel_off = &(BulgeVel[k][offset_bulge[k]]);
            bacc_off = &(BulgeAcc[k][offset_bulge[k]]);
            sort_by_id(DiskIDsChunkList[k], DiskIDsChunkListNumPer[k], DiskPart, 
                       &dpos_off, &dvel_off, &dacc_off);
            sort_by_id(BulgeIDsChunkList[k], BulgeIDsChunkListNumPer[k], BulgePart,
                       &bpos_off, &bvel_off, &bacc_off);
            offset_disk[k] += 3 * DiskIDsChunkListNumPer[k];
            offset_bulge[k] += 3 * BulgeIDsChunkListNumPer[k];
        }
        
        free(DiskPart);
        free(BulgePart);
    }

    // now loop through each ID chunk and write to a temporary file
    hsize_t disk_dims[3], bulge_dims[3], time_dims[2];
    disk_dims[0] = bulge_dims[0] = time_dims[0] = NSnapInChunk;
    disk_dims[2] = bulge_dims[2] = 3;
    time_dims[1];
    hid_t grp_disk, grp_bulge, disk_dspace_vec, bulge_dspace_vec, time_dspace, dset;
    for(j=0; j<Nchunk_id; j++){
        disk_dims[1] = DiskIDsChunkListNumPer[j];
        bulge_dims[1] = BulgeIDsChunkListNumPer[j];

        sprintf(fname, "%s/tmp%d.hdf5", prefix, j);
        file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if(file_id < 0)
            printf("UNABLE TO CREATE FILE %s", fname);

        grp_disk = H5Gcreate1(file_id, "PartType2", 0);
        grp_bulge = H5Gcreate1(file_id, "PartType3", 0);

        disk_dspace_vec = H5Screate_simple(3, disk_dims, NULL);
        bulge_dspace_vec = H5Screate_simple(3, bulge_dims, NULL);
        time_dspace = H5Screate_simple(1, time_dims, NULL);

        // Write time.
        dset = H5Dcreate1(file_id, "Time", H5T_NATIVE_DOUBLE, time_dspace, H5P_DEFAULT);
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, Time);
        H5Dclose(dset);

        // Write disk coordinates.
        dset = H5Dcreate1(grp_disk, "Coordinates", H5T_NATIVE_DOUBLE, disk_dspace_vec, H5P_DEFAULT);
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, DiskPos[j]);
        H5Dclose(dset);

        // Write disk velocities.
        dset = H5Dcreate1(grp_disk, "Velocities", H5T_NATIVE_DOUBLE, disk_dspace_vec, H5P_DEFAULT);
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, DiskVel[j]);
        H5Dclose(dset);

        // Write disk accelerations.
        dset = H5Dcreate1(grp_disk, "Acceleration", H5T_NATIVE_DOUBLE, disk_dspace_vec, H5P_DEFAULT);
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, DiskAcc[j]);
        H5Dclose(dset);

        // Write bulge coordinates.
        dset = H5Dcreate1(grp_bulge, "Coordinates", H5T_NATIVE_DOUBLE, bulge_dspace_vec, H5P_DEFAULT);
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, BulgePos[j]);
        H5Dclose(dset);

        // Write bulge velocities.
        dset = H5Dcreate1(grp_bulge, "Velocities", H5T_NATIVE_DOUBLE, bulge_dspace_vec, H5P_DEFAULT);
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, BulgeVel[j]);
        H5Dclose(dset);

        // Write bulge acclerations.
        dset = H5Dcreate1(grp_bulge, "Acceleration", H5T_NATIVE_DOUBLE, bulge_dspace_vec, H5P_DEFAULT);
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, BulgeAcc[j]);
        H5Dclose(dset);

        H5Gclose(grp_disk);
        H5Gclose(grp_bulge);
        H5Fclose(file_id);
    }

    for(int ii=0; ii<Nchunk_id; ii++){
        free(DiskPos[ii]);
        free(DiskVel[ii]);
        free(DiskAcc[ii]);
        free(BulgePos[ii]);
        free(BulgeVel[ii]);
        free(BulgeAcc[ii]);
    }
    free(DiskPos);
    free(DiskVel);
    free(DiskAcc);
    free(BulgePos);
    free(BulgeVel);
    free(BulgeAcc);
    
    free(Time);

    printf("finished with chunk %d\n", i);
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    char name[100];
    char lvl[100];
    int Nchunk_id, Nchunk_snap, Nsnap;
    char basepath[1000], output_dir[1000], fname[1000];
    int *id_chunks_disk, *indices_chunks;
    uint NumPart_Total[NTYPES];
    long long *DiskIDs, *BulgeIDs;
    int **SnapChunkList;
    int *SnapChunkListNumPer;
    long long **DiskIDsChunkList, **BulgeIDsChunkList;
    long long *DiskIDsChunkListNumPer, *BulgeIDsChunkListNumPer;
    int * SnapList;
    struct Part *DiskPart, *BulgePart;
    hid_t file_id;
    struct stat st;

    // Check to make sure right number of arguments.
    if(argc != 3){
        if(rank == 0)
            printf("Usage: ./compute_phase_space.o name lvl\n");
        exit(1);
    }

    // Copy name and lvl, print.
    strcpy(name, argv[1]);
    strcpy(lvl, argv[2]);

    if(rank == 0)
        printf("Running for name=%s, lvl=%s\n", name, lvl);

    // compute Nchunks
    compute_Nchunk(&Nchunk_id, &Nchunk_snap);
    printf("Nchunk_id=%d, Nchunk_snap=%d\n", Nchunk_id, Nchunk_snap);

    sprintf(basepath, "../../runs/%s/%s/", name, lvl);
    sprintf(output_dir, "%s/output/", basepath);

    // only do this next section on the 0th thread
    if (rank ==0)
    {
        // Write down basepath/output dir and search for the number of snapshots
        Nsnap = compute_nsnap(basepath);

        sprintf(fname, "%s/snapdir_%03d/snapshot_%03d.0.hdf5", output_dir, 0, 0);
        file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
        read_header_attribute(file_id, H5T_NATIVE_UINT, "NumPart_Total", NumPart_Total);
        read_header_attribute(file_id, H5T_NATIVE_UINT, "NumPart_Total", NumPart_Total_LastSnap);
        H5Fclose(file_id);

        // Pull out particles from the last snapshot
        get_part(output_dir, Nsnap-1, 2, &DiskPart);
        get_part(output_dir, Nsnap-1, 3, &BulgePart);

        // sort particles
        qsort(DiskPart, NumPart_Total_LastSnap[2], sizeof(struct Part), cmprID);
        qsort(BulgePart, NumPart_Total_LastSnap[3], sizeof(struct Part), cmprID);

        // construct sorted ID list
        DiskIDs = get_IDs_from_part(DiskPart, NumPart_Total_LastSnap[2]);
        BulgeIDs = get_IDs_from_part(BulgePart, NumPart_Total_LastSnap[3]);
    
        // free disk part and bulge part
        free(DiskPart);
        free(BulgePart);

        SnapList = (int *)malloc(sizeof(int) * Nsnap);
        for(int i=0; i<Nsnap; i++)
            SnapList[i] = i;
    }

    // Now broadcast from the 0th rank to all the others
    MPI_Bcast(&Nsnap, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(NumPart_Total_LastSnap, NTYPES, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // non-0 rank processes need to allocate
    if(rank != 0){
        DiskIDs = (long long *)malloc(sizeof(long long) * NumPart_Total_LastSnap[2]);
        BulgeIDs = (long long *)malloc(sizeof(long long) * NumPart_Total_LastSnap[3]);
        SnapList = (int *)malloc(sizeof(int) * Nsnap);
    }

    MPI_Bcast(DiskIDs, NumPart_Total_LastSnap[2], MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(BulgeIDs, NumPart_Total_LastSnap[3], MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(SnapList, Nsnap, MPI_INT, 0, MPI_COMM_WORLD);

    // Create snapshot chunked arrays
    array_split_int(Nchunk_snap, SnapList, Nsnap, &SnapChunkList, &SnapChunkListNumPer);

    // Create IDs chunked arrays
    array_split_llong(Nchunk_id, DiskIDs, NumPart_Total_LastSnap[2], &DiskIDsChunkList, &DiskIDsChunkListNumPer);
    array_split_llong(Nchunk_id, BulgeIDs, NumPart_Total_LastSnap[3], &BulgeIDsChunkList, &BulgeIDsChunkListNumPer);

    if(rank ==0){
        // Create output data directory if it doesn't exist
        // struct stat st = {0};
        if (stat("./data", &st) == -1) {
            mkdir("./data", 0700);
        }
        char data_dir[1000];
        sprintf(data_dir, "./data/%s-%s", name, lvl);
        if (stat(data_dir, &st) == -1) {
            mkdir(data_dir, 0700);
        }
    }

    // now we need to split the snapshot chunks into chunks across the processors (i know, confusing..)
    // but each processor has its own copy of all the snapshot chunks, so we just need to construct an array
    // of size Nchunk_snap and each processor gets its own
    // we can do this pretty easily with the array_split function from earlier
    // 
    // to make it easier to understand, we are essentially parallelizing thsi for loop:
    // for(int i=0; i<Nchunk_snap; i++){
    //     process_snap_chunk(i, output_dir, name, lvl, SnapChunkList[i], SnapChunkListNumPer[i], Nchunk_id,
    //                        DiskIDsChunkList, DiskIDsChunkListNumPer,
    //                        BulgeIDsChunkList, BulgeIDsChunkListNumPer);
    // }

    int ChunkStart, ChunkEnd;

    compute_chunk_start_end(rank, size, Nchunk_snap, &ChunkStart, &ChunkEnd);
    printf("on rank=%d, we have ChunkStart=%d and ChunkEnd=%d\n", rank, ChunkStart, ChunkEnd);

    MPI_Barrier(MPI_COMM_WORLD);

    for(int i=ChunkStart; i<ChunkEnd; i++){
        process_snap_chunk(i, output_dir, name, lvl, SnapChunkList[i], SnapChunkListNumPer[i], Nchunk_id,
                           DiskIDsChunkList, DiskIDsChunkListNumPer,
                           BulgeIDsChunkList, BulgeIDsChunkListNumPer);
    }

    // printf("DiskIDs[0]=%lld, DiskIDs[100]=%lld, DiskIDs[last]=%lld\n", DiskPart[0].ID, DiskPart[100].ID, DiskPart[NumPart_Total_LastSnap[2]-1].ID);
    // printf("BulgeIDs[0]=%lld, BulgeIDs[100]=%lld, BulgeIDs[last]=%lld\n", BulgePart[0].ID, BulgePart[100].ID, BulgePart[NumPart_Total_LastSnap[3]-1].ID);

    // printf("DiskIDs[0]=%lld, DiskIDs[100]=%lld, DiskIDs[last]=%lld\n", DiskIDs[0], DiskIDs[100], DiskIDs[NumPart_Total_LastSnap[2]-1]);
    // printf("BulgeIDs[0]=%lld, BulgeIDs[100]=%lld, BulgeIDs[last]=%lld\n", BulgeIDs[0], BulgeIDs[100], BulgeIDs[NumPart_Total_LastSnap[3]-1]);

    // printf("part[0] ID=%lld, Pos=%g|%g|%g, Vel=%g|%g|%g, Acc=%g|%g|%g\n", BulgePart[i].ID, BulgePart[i].Pos[0], BulgePart[i].Pos[1], BulgePart[i].Pos[2],
    //                                                                       BulgePart[i].Vel[0], BulgePart[i].Vel[1], BulgePart[i].Vel[2], 
    //                                                                       BulgePart[i].Acc[0], BulgePart[i].Acc[1], BulgePart[i].Acc[2]);

    // Compute nsnap

    free(SnapList);
    return 0;
}
