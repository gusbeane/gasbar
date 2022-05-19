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
int Nsnap; // total number of snapshots
int rank, size;

// Variables related to the number of chunks
int Nchunk_id, Nchunk_snap; // number of chunks to place ids and snapshots into
int **SnapChunkList;
int *SnapChunkListNumPer;
long long **HaloIDsChunkList, **DiskIDsChunkList, **BulgeIDsChunkList, **StarIDsChunkList;
long long *HaloIDsChunkListNumPer, *DiskIDsChunkListNumPer, *BulgeIDsChunkListNumPer, *StarIDsChunkListNumPer;

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

void write_Dset(hid_t loc_id, char *dname, hid_t mem_type_id, hid_t dspace, void *buf){
    hid_t dset    = H5Dcreate1(loc_id, dname, mem_type_id, dspace, H5P_DEFAULT);
    herr_t status = H5Dwrite(dset, mem_type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    
    if(status < 0){
        printf("ERROR: failed writing dataset with name %s\n", dname);
        exit(1);
    }
    
    H5Dclose(dset);
}

/*! \brief A wrapper for the printf() function.
 *
 *  This function has the same functionalities of the standard printf()
 *  function. However, data is written to the standard output only for
 *  the task with rank 0.
 *
 *  \param[in] fmt String that contains format arguments.
 *
 *  \return void
 */
void mpi_printf(const char *fmt, ...)
{
  if(rank == 0)
    {
      va_list l;
      va_start(l, fmt);
      vprintf(fmt, l);
      fflush(stdout);
      va_end(l);
    }
}

// void read_Dset(hid_t loc_id, char *dname, hid_t mem_type_id, void **buf){
//     hid_t dset = H5Dopen(loc_id, dname, H5P_DEFAULT);
//     herr_t = H5Dread(dset, mem_type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, *buf);

//     if(status < 0){
//         printf("ERROR: failed reading dataset with name %s\n", dname);
//         exit(1);
//     }

//     H5Dclose(dset);
// }

void compute_Nchunk(){
    // TODO: compute these according to memory requirements
    Nchunk_id = 1024;
    Nchunk_snap = 256;
    return;
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
    // Nsnap = 100;

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
        if(NumPart_ThisFile[PartType] > 0){
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
            free(IDs_ThisFile);
        }
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

    // printf("fname=%s, Nfiles=%d\n", fname, Nfiles);

    // printf("grp_name=%s, dset_name=%s\n", grp_name, dset_name);

    // loop through files and copy data into output
    int NumPartCum= 0;
    for(int i=0; i<Nfiles; i++){
        sprintf(fname, "%s/snapdir_%03d/snapshot_%03d.%d.hdf5", output_dir, snap_idx, snap_idx, i);
        // printf("NumPartCum=%d, reading %s\n", NumPartCum, fname);
        hid_t file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

        read_header_attribute(file_id, H5T_NATIVE_UINT, "NumPart_ThisFile", NumPart_ThisFile);
        if(NumPart_ThisFile[PartType] > 0){
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
            free(Vec_ThisFile);
        }
        H5Fclose(file_id);
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

void sort_by_id(long long *chunk_ids, long long Nids_in_chunk, struct Part * part, uint Npart, double **out_pos, double **out_vel, double **out_acc)
{
    long long itot=0;
    long long ichunk, chk_id;
    int j;

    for(ichunk=0; ichunk<Nids_in_chunk; ichunk++){
        // printf("chk_id=%lld, part[itot].ID=%lld\n", chk_id, part[itot].ID);
        chk_id = chunk_ids[ichunk];
        while(chk_id > part[itot].ID)
        {
            itot++;
            if(itot >= Npart){
                itot--;
                break;
            }
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

void process_snap_chunk(int i, char *output_dir, char *name, char *lvl){
    
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

    // pull out the snap chunk and number in the snap chunk
    int NSnapInChunk;
    int *SnapChunk;
    SnapChunk = SnapChunkList[i];
    NSnapInChunk = SnapChunkListNumPer[i];

    double *Time;
    Time = (double *)malloc(sizeof(double) * NSnapInChunk);

    double **HaloPos, **HaloVel, **HaloAcc;
    double **DiskPos, **DiskVel, **DiskAcc;
    double **BulgePos, **BulgeVel, **BulgeAcc;
    double **StarPos, **StarVel, **StarAcc;
    HaloPos = (double **)malloc(sizeof(double *) * Nchunk_id);
    HaloVel = (double **)malloc(sizeof(double *) * Nchunk_id);
    HaloAcc = (double **)malloc(sizeof(double *) * Nchunk_id);
    DiskPos = (double **)malloc(sizeof(double *) * Nchunk_id);
    DiskVel = (double **)malloc(sizeof(double *) * Nchunk_id);
    DiskAcc = (double **)malloc(sizeof(double *) * Nchunk_id);
    BulgePos = (double **)malloc(sizeof(double *) * Nchunk_id);
    BulgeVel = (double **)malloc(sizeof(double *) * Nchunk_id);
    BulgeAcc = (double **)malloc(sizeof(double *) * Nchunk_id);
    for(int ii=0; ii<Nchunk_id; ii++){
        HaloPos[ii] = (double *)malloc(sizeof(double) * NSnapInChunk * 3 * HaloIDsChunkListNumPer[ii]);
        HaloVel[ii] = (double *)malloc(sizeof(double) * NSnapInChunk * 3 * HaloIDsChunkListNumPer[ii]);
        HaloAcc[ii] = (double *)malloc(sizeof(double) * NSnapInChunk * 3 * HaloIDsChunkListNumPer[ii]);
        DiskPos[ii] = (double *)malloc(sizeof(double) * NSnapInChunk * 3 * DiskIDsChunkListNumPer[ii]);
        DiskVel[ii] = (double *)malloc(sizeof(double) * NSnapInChunk * 3 * DiskIDsChunkListNumPer[ii]);
        DiskAcc[ii] = (double *)malloc(sizeof(double) * NSnapInChunk * 3 * DiskIDsChunkListNumPer[ii]);
        BulgePos[ii] = (double *)malloc(sizeof(double) * NSnapInChunk * 3 * BulgeIDsChunkListNumPer[ii]);
        BulgeVel[ii] = (double *)malloc(sizeof(double) * NSnapInChunk * 3 * BulgeIDsChunkListNumPer[ii]);
        BulgeAcc[ii] = (double *)malloc(sizeof(double) * NSnapInChunk * 3 * BulgeIDsChunkListNumPer[ii]);
    }

    if(NumPart_Total_LastSnap[4] > 0){
        StarPos = (double **)malloc(sizeof(double *) * Nchunk_id);
        StarVel = (double **)malloc(sizeof(double *) * Nchunk_id);
        StarAcc = (double **)malloc(sizeof(double *) * Nchunk_id);
        for(int ii=0; ii<Nchunk_id; ii++){
            StarPos[ii] = (double *)malloc(sizeof(double) * NSnapInChunk * 3 * StarIDsChunkListNumPer[ii]);
            StarVel[ii] = (double *)malloc(sizeof(double) * NSnapInChunk * 3 * StarIDsChunkListNumPer[ii]);
            StarAcc[ii] = (double *)malloc(sizeof(double) * NSnapInChunk * 3 * StarIDsChunkListNumPer[ii]);
        }
    }

    // load in the snapshots
    struct Part *HaloPart, *DiskPart, *BulgePart, *StarPart;
    double this_time;
    hid_t file_id;
    long long offset_halo[Nchunk_id], offset_disk[Nchunk_id], offset_bulge[Nchunk_id], offset_star[Nchunk_id];
    for(j=0; j<Nchunk_id; j++){
        offset_halo[j] = 0;
        offset_disk[j] = 0;
        offset_bulge[j] = 0;
        offset_star[j] = 0;
    }

    double *hpos_off, *hvel_off, *hacc_off;
    double *dpos_off, *dvel_off, *dacc_off;
    double *bpos_off, *bvel_off, *bacc_off;
    double *spos_off, *svel_off, *sacc_off;
    for(j=0; j<NSnapInChunk; j++){
        // read necessary header attributes
        sprintf(fname, "%s/snapdir_%03d/snapshot_%03d.0.hdf5", output_dir, SnapChunk[j], SnapChunk[j]);
        file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
        read_header_attribute(file_id, H5T_NATIVE_UINT, "NumPart_Total", NumPart_Total);
        read_header_attribute(file_id, H5T_NATIVE_DOUBLE, "Time", &this_time);
        H5Fclose(file_id);
        Time[j] = this_time;
        printf("processing snap %d (%d of %d), time=%g, rank=%d\n", SnapChunk[j], j, NSnapInChunk-1, Time[j], rank);

        // load in the snapshots
        get_part(output_dir, SnapChunk[j], 1, &HaloPart);
        get_part(output_dir, SnapChunk[j], 2, &DiskPart);
        get_part(output_dir, SnapChunk[j], 3, &BulgePart);
        if(NumPart_Total[4] > 0){
            get_part(output_dir, SnapChunk[j], 4, &StarPart);
        }

        // printf("rank=%d got to before qsort\n", rank);

        // sort by ID
        qsort(HaloPart, NumPart_Total[1], sizeof(struct Part), cmprID);
        qsort(DiskPart, NumPart_Total[2], sizeof(struct Part), cmprID);
        qsort(BulgePart, NumPart_Total[3], sizeof(struct Part), cmprID);
        if(NumPart_Total[4] > 0){
            qsort(StarPart, NumPart_Total[4], sizeof(struct Part), cmprID);
        }

        // printf("rank=%d got to after qsort\n", rank);

        // now loop through chunks and sort pos and vel into output
        for(int k=0; k<Nchunk_id; k++){
            // printf("rank=%d got to 0\n", rank);
            hpos_off = &(HaloPos[k][offset_halo[k]]);
            hvel_off = &(HaloVel[k][offset_halo[k]]);
            hacc_off = &(HaloAcc[k][offset_halo[k]]);
            dpos_off = &(DiskPos[k][offset_disk[k]]);
            dvel_off = &(DiskVel[k][offset_disk[k]]);
            dacc_off = &(DiskAcc[k][offset_disk[k]]);
            bpos_off = &(BulgePos[k][offset_bulge[k]]);
            bvel_off = &(BulgeVel[k][offset_bulge[k]]);
            bacc_off = &(BulgeAcc[k][offset_bulge[k]]);
            // printf("rank=%d got to 1\n", rank);
            sort_by_id(HaloIDsChunkList[k], HaloIDsChunkListNumPer[k], HaloPart, NumPart_Total[1],
                       &hpos_off, &hvel_off, &hacc_off);
            sort_by_id(DiskIDsChunkList[k], DiskIDsChunkListNumPer[k], DiskPart, NumPart_Total[2],
                       &dpos_off, &dvel_off, &dacc_off);
            sort_by_id(BulgeIDsChunkList[k], BulgeIDsChunkListNumPer[k], BulgePart, NumPart_Total[3],
                       &bpos_off, &bvel_off, &bacc_off);
            // printf("rank=%d got to 2\n", rank);
            

            if(NumPart_Total[4] > 0){
                spos_off = &(StarPos[k][offset_star[k]]);
                svel_off = &(StarVel[k][offset_star[k]]);
                sacc_off = &(StarAcc[k][offset_star[k]]);
                sort_by_id(StarIDsChunkList[k], StarIDsChunkListNumPer[k], StarPart, NumPart_Total[4],
                       &spos_off, &svel_off, &sacc_off);
            }

            offset_halo[k] += 3 * HaloIDsChunkListNumPer[k];
            offset_disk[k] += 3 * DiskIDsChunkListNumPer[k];
            offset_bulge[k] += 3 * BulgeIDsChunkListNumPer[k];
            offset_star[k] += 3 * StarIDsChunkListNumPer[k];

        }

        // printf("rank=%d got to after chunk loop\n", rank);


        free(HaloPart);
        free(DiskPart);
        free(BulgePart);
        if(NumPart_Total[4] > 0)
            free(StarPart);
    }

    // if we have star particles in last snap but none in this snap, we have to initialize
    // all the star particle arrays as nans
    if((NumPart_Total_LastSnap[4] > 0) && (NumPart_Total[4] == 0)){
        for(int ii=0; ii<Nchunk_id; ii++){
            for(j=0; j<NSnapInChunk * 3 * StarIDsChunkListNumPer[ii]; j++){
                StarPos[ii][j] = NAN;
                StarVel[ii][j] = NAN;
                StarAcc[ii][j] = NAN;
            }
        }
    }

    // now loop through each ID chunk and write to a temporary file
    hsize_t halo_dims[3], disk_dims[3], bulge_dims[3], star_dims[3], time_dims[2];
    halo_dims[0] = disk_dims[0] = bulge_dims[0] = star_dims[0] = time_dims[0] = NSnapInChunk;
    halo_dims[2] = disk_dims[2] = bulge_dims[2] = star_dims[2] = 3;
    time_dims[1] = 1;
    hid_t grp_halo, grp_disk, grp_bulge, grp_star;
    hid_t halo_dspace_vec, disk_dspace_vec, bulge_dspace_vec, star_dspace_vec, time_dspace, dset;
    for(j=0; j<Nchunk_id; j++){
        halo_dims[1] = HaloIDsChunkListNumPer[j];
        disk_dims[1] = DiskIDsChunkListNumPer[j];
        bulge_dims[1] = BulgeIDsChunkListNumPer[j];
        star_dims[1] = StarIDsChunkListNumPer[j];

        sprintf(fname, "%s/tmp%d.hdf5", prefix, j);
        file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if(file_id < 0)
            printf("UNABLE TO CREATE FILE %s", fname);

        grp_halo = H5Gcreate1(file_id, "PartType1", 0);
        grp_disk = H5Gcreate1(file_id, "PartType2", 0);
        grp_bulge = H5Gcreate1(file_id, "PartType3", 0);

        halo_dspace_vec = H5Screate_simple(3, halo_dims, NULL);
        disk_dspace_vec = H5Screate_simple(3, disk_dims, NULL);
        bulge_dspace_vec = H5Screate_simple(3, bulge_dims, NULL);
        star_dspace_vec = H5Screate_simple(3, star_dims, NULL);
        time_dspace = H5Screate_simple(1, time_dims, NULL);

        // Write time.
        write_Dset(file_id, "Time", H5T_NATIVE_DOUBLE, time_dspace, Time);

        // Write disk output.
        write_Dset(grp_halo, "Coordinates", H5T_NATIVE_DOUBLE, halo_dspace_vec, HaloPos[j]);
        write_Dset(grp_halo, "Velocities", H5T_NATIVE_DOUBLE, halo_dspace_vec, HaloVel[j]);
        write_Dset(grp_halo, "Acceleration", H5T_NATIVE_DOUBLE, halo_dspace_vec, HaloAcc[j]);

        // Write disk output.
        write_Dset(grp_disk, "Coordinates", H5T_NATIVE_DOUBLE, disk_dspace_vec, DiskPos[j]);
        write_Dset(grp_disk, "Velocities", H5T_NATIVE_DOUBLE, disk_dspace_vec, DiskVel[j]);
        write_Dset(grp_disk, "Acceleration", H5T_NATIVE_DOUBLE, disk_dspace_vec, DiskAcc[j]);

        // Write bulge output.
        write_Dset(grp_bulge, "Coordinates", H5T_NATIVE_DOUBLE, bulge_dspace_vec, BulgePos[j]);
        write_Dset(grp_bulge, "Velocities", H5T_NATIVE_DOUBLE, bulge_dspace_vec, BulgeVel[j]);
        write_Dset(grp_bulge, "Acceleration", H5T_NATIVE_DOUBLE, bulge_dspace_vec, BulgeAcc[j]);

        // Write star output (if needed).
        if(NumPart_Total_LastSnap[4] > 0){
            grp_star = H5Gcreate1(file_id, "PartType4", 0);
            write_Dset(grp_star, "Coordinates", H5T_NATIVE_DOUBLE, star_dspace_vec, StarPos[j]);
            write_Dset(grp_star, "Velocities", H5T_NATIVE_DOUBLE, star_dspace_vec, StarVel[j]);
            write_Dset(grp_star, "Acceleration", H5T_NATIVE_DOUBLE, star_dspace_vec, StarAcc[j]);
            H5Gclose(grp_star);
        }

        // Close hdf5 stuff.
        H5Gclose(grp_halo);
        H5Gclose(grp_disk);
        H5Gclose(grp_bulge);
        H5Fclose(file_id);
    }

    // Free everything.
    for(int ii=0; ii<Nchunk_id; ii++){
        free(HaloPos[ii]);
        free(HaloVel[ii]);
        free(HaloAcc[ii]);
        free(DiskPos[ii]);
        free(DiskVel[ii]);
        free(DiskAcc[ii]);
        free(BulgePos[ii]);
        free(BulgeVel[ii]);
        free(BulgeAcc[ii]);
        if(NumPart_Total_LastSnap[4] > 0){
            free(StarPos[ii]);
            free(StarVel[ii]);
            free(StarAcc[ii]);
        }
    }
    free(HaloPos);
    free(HaloVel);
    free(HaloAcc);
    free(DiskPos);
    free(DiskVel);
    free(DiskAcc);
    free(BulgePos);
    free(BulgeVel);
    free(BulgeAcc);
    if(NumPart_Total_LastSnap[4] > 0){
        free(StarPos);
        free(StarVel);
        free(StarAcc);
    }
    
    free(Time);

    printf("finished with chunk %d on rank=%d\n", i, rank);
}

void process_id_chunk(int i, char *name, char *lvl){
    int j;
    char data_dir[1000], fname[1000];
    sprintf(data_dir, "./data/%s-%s/", name, lvl);
    hid_t file_id, grp_id, dset;

    long long *HaloIDsChunk, *DiskIDsChunk, *BulgeIDsChunk, *StarIDsChunk;
    long long HaloIDsChunkNum, DiskIDsChunkNum, BulgeIDsChunkNum, StarIDsChunkNum;
    HaloIDsChunk = HaloIDsChunkList[i];
    DiskIDsChunk = DiskIDsChunkList[i];
    BulgeIDsChunk = BulgeIDsChunkList[i];
    if(NumPart_Total_LastSnap[4]>0)
        StarIDsChunk = StarIDsChunkList[i];

    HaloIDsChunkNum = HaloIDsChunkListNumPer[i];
    DiskIDsChunkNum = DiskIDsChunkListNumPer[i];
    BulgeIDsChunkNum = BulgeIDsChunkListNumPer[i];
    StarIDsChunkNum = StarIDsChunkListNumPer[i];

    double *HaloPos, *HaloVel, *HaloAcc;
    double *DiskPos, *DiskVel, *DiskAcc;
    double *BulgePos, *BulgeVel, *BulgeAcc;
    double *StarPos, *StarVel, *StarAcc;
    double *Time;

    Time = (double *)malloc(sizeof(double) * Nsnap);
    HaloPos = (double *)malloc(sizeof(double) * Nsnap * HaloIDsChunkNum * 3);
    HaloVel = (double *)malloc(sizeof(double) * Nsnap * HaloIDsChunkNum * 3);
    HaloAcc = (double *)malloc(sizeof(double) * Nsnap * HaloIDsChunkNum * 3);
    DiskPos = (double *)malloc(sizeof(double) * Nsnap * DiskIDsChunkNum * 3);
    DiskVel = (double *)malloc(sizeof(double) * Nsnap * DiskIDsChunkNum * 3);
    DiskAcc = (double *)malloc(sizeof(double) * Nsnap * DiskIDsChunkNum * 3);
    BulgePos = (double *)malloc(sizeof(double) * Nsnap * BulgeIDsChunkNum * 3);
    BulgeVel = (double *)malloc(sizeof(double) * Nsnap * BulgeIDsChunkNum * 3);
    BulgeAcc = (double *)malloc(sizeof(double) * Nsnap * BulgeIDsChunkNum * 3);
    if(NumPart_Total_LastSnap[4] > 0){
        StarPos = (double *)malloc(sizeof(double) * Nsnap * StarIDsChunkNum * 3);
        StarVel = (double *)malloc(sizeof(double) * Nsnap * StarIDsChunkNum * 3);
        StarAcc = (double *)malloc(sizeof(double) * Nsnap * StarIDsChunkNum * 3);
    }

    int Ncum_time = 0;
    long long Ncum_Halo, Ncum_Disk, Ncum_Bulge, Ncum_Star;
    Ncum_Halo = Ncum_Disk = Ncum_Bulge = Ncum_Star = 0;
    for(j=0; j<Nchunk_snap; j++){
        // j is snapshot chunk idx, i is id chunk idx
        sprintf(fname, "%s/tmp%d/tmp%d.hdf5", data_dir, j, i);

        file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
        
        // read time
        dset = H5Dopen(file_id, "Time", H5P_DEFAULT);
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(Time[Ncum_time]));
        H5Dclose(dset);


        grp_id = H5Gopen(file_id, "PartType1", H5P_DEFAULT);
        // read disk pos
        dset = H5Dopen(grp_id, "Coordinates", H5P_DEFAULT);
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(HaloPos[Ncum_Halo]));
        H5Dclose(dset);

        dset = H5Dopen(grp_id, "Velocities", H5P_DEFAULT);
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(HaloVel[Ncum_Halo]));
        H5Dclose(dset);

        dset = H5Dopen(grp_id, "Acceleration", H5P_DEFAULT);
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(HaloAcc[Ncum_Halo]));
        H5Dclose(dset);
        H5Gclose(grp_id);


        grp_id = H5Gopen(file_id, "PartType2", H5P_DEFAULT);
        // read disk pos
        dset = H5Dopen(grp_id, "Coordinates", H5P_DEFAULT);
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(DiskPos[Ncum_Disk]));
        H5Dclose(dset);

        dset = H5Dopen(grp_id, "Velocities", H5P_DEFAULT);
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(DiskVel[Ncum_Disk]));
        H5Dclose(dset);

        dset = H5Dopen(grp_id, "Acceleration", H5P_DEFAULT);
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(DiskAcc[Ncum_Disk]));
        H5Dclose(dset);
        H5Gclose(grp_id);

        grp_id = H5Gopen(file_id, "PartType3", H5P_DEFAULT);
        // read disk pos
        dset = H5Dopen(grp_id, "Coordinates", H5P_DEFAULT);
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(BulgePos[Ncum_Bulge]));
        H5Dclose(dset);

        dset = H5Dopen(grp_id, "Velocities", H5P_DEFAULT);
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(BulgeVel[Ncum_Bulge]));
        H5Dclose(dset);

        dset = H5Dopen(grp_id, "Acceleration", H5P_DEFAULT);
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(BulgeAcc[Ncum_Bulge]));
        H5Dclose(dset);
        H5Gclose(grp_id);

        if(NumPart_Total_LastSnap[4] > 0){
            grp_id = H5Gopen(file_id, "PartType4", H5P_DEFAULT);
            // read disk pos
            dset = H5Dopen(grp_id, "Coordinates", H5P_DEFAULT);
            H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(StarPos[Ncum_Star]));
            H5Dclose(dset);

            dset = H5Dopen(grp_id, "Velocities", H5P_DEFAULT);
            H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(StarVel[Ncum_Star]));
            H5Dclose(dset);

            dset = H5Dopen(grp_id, "Acceleration", H5P_DEFAULT);
            H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(StarAcc[Ncum_Star]));
            H5Dclose(dset);
            H5Gclose(grp_id);
        }

        H5Fclose(file_id);

        Ncum_time += SnapChunkListNumPer[j];
        Ncum_Halo += 3 * HaloIDsChunkNum * SnapChunkListNumPer[j];
        Ncum_Disk += 3 * DiskIDsChunkNum * SnapChunkListNumPer[j];
        Ncum_Bulge += 3 * BulgeIDsChunkNum * SnapChunkListNumPer[j];
        Ncum_Star += 3 * StarIDsChunkNum * SnapChunkListNumPer[j];
    }

    // now write to output file
    sprintf(fname, "%s/phase_space_%s-%s.%d.hdf5", data_dir, name, lvl, i);

    hsize_t halo_dims[3], disk_dims[3], bulge_dims[3], star_dims[3], time_dims[2];
    hsize_t halo_id_dims[2], disk_id_dims[2], bulge_id_dims[2], star_id_dims[2];
    halo_dims[0] = disk_dims[0] = bulge_dims[0] = star_dims[0] = time_dims[0] = Nsnap;
    halo_dims[2] = disk_dims[2] = bulge_dims[2] = star_dims[2] = 3;
    time_dims[1] = 1;
    hid_t grp_halo, grp_disk, grp_bulge, grp_star;
    hid_t halo_dspace_vec, disk_dspace_vec, bulge_dspace_vec, star_dspace_vec, time_dspace;
    hid_t halo_dspace_ids, disk_dspace_ids, bulge_dspace_ids, star_dspace_ids;
    
    halo_dims[1] = HaloIDsChunkNum;
    disk_dims[1] = DiskIDsChunkNum;
    bulge_dims[1] = BulgeIDsChunkNum;
    star_dims[1] = StarIDsChunkNum;

    halo_id_dims[0] = HaloIDsChunkNum;
    disk_id_dims[0] = DiskIDsChunkNum;
    bulge_id_dims[0] = BulgeIDsChunkNum;
    star_id_dims[0] = StarIDsChunkNum;
    halo_id_dims[1] = disk_id_dims[1] = bulge_id_dims[1] = star_id_dims[1] = 1;

    file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if(file_id < 0)
        printf("UNABLE TO CREATE FILE %s", fname);

    grp_halo = H5Gcreate1(file_id, "PartType1", 0);
    grp_disk = H5Gcreate1(file_id, "PartType2", 0);
    grp_bulge = H5Gcreate1(file_id, "PartType3", 0);
    if(NumPart_Total_LastSnap[4] > 0)
        grp_star = H5Gcreate1(file_id, "PartType4", 0);

    halo_dspace_vec = H5Screate_simple(3, halo_dims, NULL);
    disk_dspace_vec = H5Screate_simple(3, disk_dims, NULL);
    bulge_dspace_vec = H5Screate_simple(3, bulge_dims, NULL);
    star_dspace_vec = H5Screate_simple(3, star_dims, NULL);
    
    halo_dspace_ids = H5Screate_simple(1, halo_id_dims, NULL);
    disk_dspace_ids = H5Screate_simple(1, disk_id_dims, NULL);
    bulge_dspace_ids = H5Screate_simple(1, bulge_id_dims, NULL);
    star_dspace_ids = H5Screate_simple(1, star_id_dims, NULL);
    time_dspace = H5Screate_simple(1, time_dims, NULL);

    // Write time.
    write_Dset(file_id, "Time", H5T_NATIVE_DOUBLE, time_dspace, Time);
    
    // Write disk stuff.
    write_Dset(grp_halo, "ParticleIDs", H5T_NATIVE_LLONG, halo_dspace_ids, HaloIDsChunk);
    write_Dset(grp_halo, "Coordinates", H5T_NATIVE_DOUBLE, halo_dspace_vec, HaloPos);
    write_Dset(grp_halo, "Velocities", H5T_NATIVE_DOUBLE, halo_dspace_vec, HaloVel);
    write_Dset(grp_halo, "Acceleration", H5T_NATIVE_DOUBLE, halo_dspace_vec, HaloAcc);

    // Write disk stuff.
    write_Dset(grp_disk, "ParticleIDs", H5T_NATIVE_LLONG, disk_dspace_ids, DiskIDsChunk);
    write_Dset(grp_disk, "Coordinates", H5T_NATIVE_DOUBLE, disk_dspace_vec, DiskPos);
    write_Dset(grp_disk, "Velocities", H5T_NATIVE_DOUBLE, disk_dspace_vec, DiskVel);
    write_Dset(grp_disk, "Acceleration", H5T_NATIVE_DOUBLE, disk_dspace_vec, DiskAcc);

    // Write bulge stuff.
    write_Dset(grp_bulge, "ParticleIDs", H5T_NATIVE_LLONG, bulge_dspace_ids, BulgeIDsChunk);
    write_Dset(grp_bulge, "Coordinates", H5T_NATIVE_DOUBLE, bulge_dspace_vec, BulgePos);
    write_Dset(grp_bulge, "Velocities", H5T_NATIVE_DOUBLE, bulge_dspace_vec, BulgeVel);
    write_Dset(grp_bulge, "Acceleration", H5T_NATIVE_DOUBLE, bulge_dspace_vec, BulgeAcc);

    // Write star stuff (if needed).
    if(NumPart_Total_LastSnap[4] > 0){
        write_Dset(grp_star, "ParticleIDs", H5T_NATIVE_LLONG, star_dspace_ids, StarIDsChunk);
        write_Dset(grp_star, "Coordinates", H5T_NATIVE_DOUBLE, star_dspace_vec, StarPos);
        write_Dset(grp_star, "Velocities", H5T_NATIVE_DOUBLE, star_dspace_vec, StarVel);
        write_Dset(grp_star, "Acceleration", H5T_NATIVE_DOUBLE, star_dspace_vec, StarAcc);
        H5Gclose(grp_star);
    }

    // Close hdf5 stuff.
    H5Gclose(grp_halo);
    H5Gclose(grp_disk);
    H5Gclose(grp_bulge);
    H5Fclose(file_id);

    // Free everything.
    free(HaloPos);
    free(HaloVel);
    free(HaloAcc);
    free(DiskPos);
    free(DiskVel);
    free(DiskAcc);
    free(BulgePos);
    free(BulgeVel);
    free(BulgeAcc);
    if(NumPart_Total_LastSnap[4] > 0){
        free(StarPos);
        free(StarVel);
        free(StarAcc);
    }
    
    free(Time);

}

void construct_id_snap_chunks(char *output_dir)
{

    fflush(stdout);
    int * SnapList;
    long long *HaloIDs, *DiskIDs, *BulgeIDs, *StarIDs;
    char fname[1000];
    hid_t file_id;
    struct Part *HaloPart, *DiskPart, *BulgePart, *StarPart;

    


    // only do this section on the 0th thread
    if (rank ==0)
    {
        // search for the number of snapshots
        compute_Nsnap(output_dir);
        Nchunk_snap = Nsnap;


        sprintf(fname, "%s/snapdir_%03d/snapshot_%03d.0.hdf5", output_dir, Nsnap-1, Nsnap-1);
        file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
        read_header_attribute(file_id, H5T_NATIVE_UINT, "NumPart_Total", NumPart_Total_LastSnap);
        H5Fclose(file_id);

        // Pull out particles from the last snapshot
        get_part(output_dir, Nsnap-1, 1, &HaloPart);
        get_part(output_dir, Nsnap-1, 2, &DiskPart);
        get_part(output_dir, Nsnap-1, 3, &BulgePart);

        // sort particles
        qsort(HaloPart, NumPart_Total_LastSnap[1], sizeof(struct Part), cmprID);
        qsort(DiskPart, NumPart_Total_LastSnap[2], sizeof(struct Part), cmprID);
        qsort(BulgePart, NumPart_Total_LastSnap[3], sizeof(struct Part), cmprID);


        // construct sorted ID list
        HaloIDs = get_IDs_from_part(HaloPart, NumPart_Total_LastSnap[1]);
        DiskIDs = get_IDs_from_part(DiskPart, NumPart_Total_LastSnap[2]);
        BulgeIDs = get_IDs_from_part(BulgePart, NumPart_Total_LastSnap[3]);
    
        // free disk part and bulge part
        free(HaloPart);
        free(DiskPart);
        free(BulgePart);

        // if stars exist, do the same steps again
        if(NumPart_Total_LastSnap[4] > 0){
            get_part(output_dir, Nsnap-1, 4, &StarPart);
            qsort(StarPart, NumPart_Total_LastSnap[4], sizeof(struct Part), cmprID);
            StarIDs = get_IDs_from_part(StarPart, NumPart_Total_LastSnap[4]);
            free(StarPart);
        }

        // Construct snapshot list
        SnapList = (int *)malloc(sizeof(int) * Nsnap);
        for(int i=0; i<Nsnap; i++)
            SnapList[i] = i;
    }


    // Now broadcast from the 0th rank to all the others
    MPI_Bcast(&Nsnap, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Nchunk_snap, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(NumPart_Total_LastSnap, NTYPES, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // non-0 rank processes need to allocate
    if(rank != 0){
        HaloIDs = (long long *)malloc(sizeof(long long) * NumPart_Total_LastSnap[1]);
        DiskIDs = (long long *)malloc(sizeof(long long) * NumPart_Total_LastSnap[2]);
        BulgeIDs = (long long *)malloc(sizeof(long long) * NumPart_Total_LastSnap[3]);
        SnapList = (int *)malloc(sizeof(int) * Nsnap);
        if(NumPart_Total_LastSnap[4] > 0)
            StarIDs = (long long *)malloc(sizeof(long long) * NumPart_Total_LastSnap[4]);
    }


    MPI_Bcast(HaloIDs, NumPart_Total_LastSnap[1], MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(DiskIDs, NumPart_Total_LastSnap[2], MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(BulgeIDs, NumPart_Total_LastSnap[3], MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    if(NumPart_Total_LastSnap[4] > 0)
        MPI_Bcast(StarIDs, NumPart_Total_LastSnap[4], MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(SnapList, Nsnap, MPI_INT, 0, MPI_COMM_WORLD);

    // Create snapshot chunked arrays
    array_split_int(Nchunk_snap, SnapList, Nsnap, &SnapChunkList, &SnapChunkListNumPer);

    // Create IDs chunked arrays
    array_split_llong(Nchunk_id, HaloIDs, NumPart_Total_LastSnap[1], &HaloIDsChunkList, &HaloIDsChunkListNumPer);
    array_split_llong(Nchunk_id, DiskIDs, NumPart_Total_LastSnap[2], &DiskIDsChunkList, &DiskIDsChunkListNumPer);
    array_split_llong(Nchunk_id, BulgeIDs, NumPart_Total_LastSnap[3], &BulgeIDsChunkList, &BulgeIDsChunkListNumPer);
    if(NumPart_Total_LastSnap[4] > 0)
        array_split_llong(Nchunk_id, StarIDs, NumPart_Total_LastSnap[4], &StarIDsChunkList, &StarIDsChunkListNumPer);
    else{
        StarIDsChunkListNumPer = (long long *)malloc(sizeof(long long) * Nchunk_id);
        for(int ii=0; ii<Nchunk_id; ii++){
            StarIDsChunkListNumPer[ii] = 0;
        }
    }

    // Now free DiskIDs and BulgeIDs, no longer needed
    free(HaloIDs);
    free(DiskIDs);
    free(BulgeIDs);
    free(SnapList);
    if(NumPart_Total_LastSnap[4] > 0)
        free(StarIDs);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    char name[100];
    char lvl[100];
    char basepath[1000], output_dir[1000], fname[1000];
    int *id_chunks_disk, *indices_chunks;
    uint NumPart_Total[NTYPES];
    long long *DiskIDs, *BulgeIDs;
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

    mpi_printf("Running for name=%s, lvl=%s\n", name, lvl);

    // compute Nchunks
    compute_Nchunk();
    mpi_printf("Nchunk_id=%d, Nchunk_snap=%d\n", Nchunk_id, Nchunk_snap);

    sprintf(basepath, "../../runs/%s/%s/", name, lvl);
    sprintf(output_dir, "%s/output/", basepath);
    
    construct_id_snap_chunks(output_dir);

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
    int ChunkStart, ChunkEnd;

    compute_chunk_start_end(rank, size, Nchunk_snap, &ChunkStart, &ChunkEnd);
    printf("on rank=%d, we have ChunkStart=%d and ChunkEnd=%d\n", rank, ChunkStart, ChunkEnd);

    MPI_Barrier(MPI_COMM_WORLD);

    // loop through chunks of snapshots and write to temporary output files
    for(int i=ChunkStart; i<ChunkEnd; i++){
        process_snap_chunk(i, output_dir, name, lvl);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // now loop through chunks of ids and read temporary output files, concatting into final output files
    compute_chunk_start_end(rank, size, Nchunk_id, &ChunkStart, &ChunkEnd);

    for(int i=ChunkStart; i<ChunkEnd; i++){
        process_id_chunk(i, name, lvl);
    }

    return 0;
}
