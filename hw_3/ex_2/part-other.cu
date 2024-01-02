#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 32
#define CUDA_CALL(F)                                    \
    if ((F) != cudaSuccess)                             \
    {                                                   \
        printf("Error at line %d: %s\n", __LINE__,      \
               cudaGetErrorString(cudaGetLastError())); \
        exit(-1);                                       \
    };
void cudaMemcpyParticlesHostToDevice(particles **const d_particle, const particles *const h_particle)
{
    /*
    INPUT SPECIFICATIONS:
    d_particle: a pointer to a particle instance that should have all variable fields NOT allocated on the host or the device.
    h_particle: a pointer to a particle instance that should have all variable fields  pre-allocated on the host with malloc or new. The fields with pointers i.e. x,y,z,u,v,w should be directed to a host array.
    MODIFICATIONS:
    d_particle: The particle instance it is pointing to will have all variable fields  allocated on the device with cudaMalloc. The fields with pointers i.e. x,y,z,u,v,w should be directed to a device array, that has been allocated separately.
    FUNCTIONALITY:
    Copy the h_particle to device as d_particle, while allocating memory for d_particle.
    */

    // Shallow Copy a h_particle
    particles copyParticle = *h_particle;
    particles *const h_particle_copy = &copyParticle;

    // Define the arrays needed to be copied to device
    FPpart *d_x;
    FPpart *d_y;
    FPpart *d_z;
    FPpart *d_u;
    FPpart *d_v;
    FPpart *d_w;

    // Allocate the memory for these devices
    CUDA_CALL(cudaMalloc((void **)d_particle, sizeof(particles)));
    CUDA_CALL(cudaMalloc((void **)&d_x, sizeof(FPpart) * h_particle->npmax));
    CUDA_CALL(cudaMalloc((void **)&d_y, sizeof(FPpart) * h_particle->npmax));
    CUDA_CALL(cudaMalloc((void **)&d_z, sizeof(FPpart) * h_particle->npmax));
    CUDA_CALL(cudaMalloc((void **)&d_u, sizeof(FPpart) * h_particle->npmax));
    CUDA_CALL(cudaMalloc((void **)&d_v, sizeof(FPpart) * h_particle->npmax));
    CUDA_CALL(cudaMalloc((void **)&d_w, sizeof(FPpart) * h_particle->npmax));

    // Copy the arrays to the device
    CUDA_CALL(cudaMemcpy(d_x, h_particle->x, sizeof(FPpart) * h_particle->npmax, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_y, h_particle->y, sizeof(FPpart) * h_particle->npmax, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_z, h_particle->z, sizeof(FPpart) * h_particle->npmax, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_u, h_particle->u, sizeof(FPpart) * h_particle->npmax, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_v, h_particle->v, sizeof(FPpart) * h_particle->npmax, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_w, h_particle->w, sizeof(FPpart) * h_particle->npmax, cudaMemcpyHostToDevice));

    // Redirect pointers in h_particle_copy to devices
    h_particle_copy->x = d_x;
    h_particle_copy->y = d_y;
    h_particle_copy->z = d_z;
    h_particle_copy->u = d_u;
    h_particle_copy->v = d_v;
    h_particle_copy->w = d_w;

    // Copy the h_particle_copy struct to the device
    CUDA_CALL(cudaMemcpy(*d_particle, h_particle_copy, sizeof(particles), cudaMemcpyHostToDevice));
}

void cudaMemcpyParticlesDeviceToHost(particles *const h_particle, particles **const d_particle)
{
    /*
    INPUT SPECIFICATIONS:
    d_particle: a pointer to a particle instance with all variable fields allocated on the device with cudaMalloc. The fields with pointers i.e. x,y,z,u,v,w should be directed to a device array, that has been allocated separately.
    h_particle: a pointer to a particle instance that should have all variable fields  pre-allocated on the host with malloc or new. The fields with pointers i.e. x,y,z,u,v,w should be directed to a host array.
    MODIFICATIONS:
    d_particle: The particle instance it is pointing to will be deep cleaned.
    h_particle: The particle instance it is pointing to will become overwritten by another one with the same specification.
    FUNCTIONALITY:
    Copy the d_particle to host as h_particle, while deallocating memory for d_particle.
    */

    // Save a shallow copy of h_particle, all pointers in there are allocated to host memory
    particles h_particle_copy_instance = *h_particle;
    particles *h_particle_copy = &h_particle_copy_instance;

    // Copy the d_particle from device to host
    CUDA_CALL(cudaMemcpy(h_particle, *d_particle, sizeof(particles), cudaMemcpyDeviceToHost));

    // Free the d_particle pointer
    CUDA_CALL(cudaFree(*d_particle));

    // Copy the pointers from device to host
    CUDA_CALL(cudaMemcpy(h_particle_copy->x, h_particle->x, sizeof(int) * h_particle->npmax, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_particle_copy->y, h_particle->y, sizeof(int) * h_particle->npmax, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_particle_copy->z, h_particle->z, sizeof(int) * h_particle->npmax, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_particle_copy->u, h_particle->u, sizeof(int) * h_particle->npmax, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_particle_copy->v, h_particle->v, sizeof(int) * h_particle->npmax, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_particle_copy->w, h_particle->w, sizeof(int) * h_particle->npmax, cudaMemcpyDeviceToHost));

    // Free the x,y,z,u,v,w pointers
    CUDA_CALL(cudaFree(h_particle->x));
    CUDA_CALL(cudaFree(h_particle->y));
    CUDA_CALL(cudaFree(h_particle->z));
    CUDA_CALL(cudaFree(h_particle->u));
    CUDA_CALL(cudaFree(h_particle->v));
    CUDA_CALL(cudaFree(h_particle->w));

    // Redirect the h_particle pointer to a host array
    h_particle->x = h_particle_copy->x;
    h_particle->y = h_particle_copy->y;
    h_particle->z = h_particle_copy->z;
    h_particle->u = h_particle_copy->u;
    h_particle->v = h_particle_copy->v;
    h_particle->w = h_particle_copy->w;
}

void cudaMemcpyGridHostToDevice(grid **const d_grid, const grid *const h_grid)
{
    /*
    INPUT SPECIFICATIONS:
    d_grid: a pointer to a grid instance that should have all variable fields NOT allocated on the host or the device.
    h_grid: a pointer to a grid instance that should have all variable fields  pre-allocated on the host with malloc or new. The fields with either 1D pointers i.e. XN_flat,YN_flat,ZN_flat or 3D pointers i.e. XN,YN,ZN should be directed to host arrays.
    MODIFICATIONS:
    d_grid: The grid instance it is pointing to will have all variable fields allocated on the device with cudaMalloc. The fields with 1D pointers i.e. XN_flat,YN_flat,ZN_flat should be directed to a device array, that has been allocated separately. The fields with 3D pointers i.e. XN,YN,ZN should be UNDEFINED.
    FUNCTIONALITY:
    Copy the h_grid to device as d_grid, while allocating memory for d_grid.
    */
    // Shallow Copy a h_grid
    grid copyGrid = *h_grid;
    grid *const h_grid_copy = &copyGrid;

    // Define the arrays needed to be copied to device
    FPfield *d_XN_flat;
    FPfield *d_YN_flat;
    FPfield *d_ZN_flat;

    // Allocate the memory for these devices
    CUDA_CALL(cudaMalloc((void **)d_grid, sizeof(grid)));
    size_t arraySize = sizeof(FPfield) * h_grid->nxn * h_grid->nyn * h_grid->nzn;
    CUDA_CALL(cudaMalloc((void **)&d_XN_flat, arraySize));
    CUDA_CALL(cudaMalloc((void **)&d_YN_flat, arraySize));
    CUDA_CALL(cudaMalloc((void **)&d_ZN_flat, arraySize));

    // Copy the arrays to the device
    CUDA_CALL(cudaMemcpy(d_XN_flat, h_grid->XN_flat, arraySize, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_YN_flat, h_grid->YN_flat, arraySize, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_ZN_flat, h_grid->ZN_flat, arraySize, cudaMemcpyHostToDevice));

    // Redirect pointers in h_grid_copy to devices
    h_grid_copy->XN_flat = d_XN_flat;
    h_grid_copy->YN_flat = d_YN_flat;
    h_grid_copy->ZN_flat = d_ZN_flat;

    // Copy the h_grid_copy struct to the device
    CUDA_CALL(cudaMemcpy(*d_grid, h_grid_copy, sizeof(grid), cudaMemcpyHostToDevice));
}

void cudaFreeGrid(grid **const d_grid)
{
    /*
    INPUT SPECIFICATIONS:
    h_grid: a pointer to a grid instance that should have all variable fields pre-allocated on the host with malloc or new. The fields with either 1D pointers i.e. XN_flat,YN_flat,ZN_flat or 3D pointers i.e. XN,YN,ZN should be directed to host array.
    MODIFICATIONS:
    d_grid: The grid instance it is pointing to will be deep cleaned.
    FUNCTIONALITY:
    Deallocate memory for d_grid.
    */

    // Save a copy of h_grid, all pointers in there are allocated to host memory
    grid h_grid_instance;
    grid *h_grid = &h_grid_instance;

    // Copy the d_grid from device to host
    CUDA_CALL(cudaMemcpy(h_grid, *d_grid, sizeof(grid), cudaMemcpyDeviceToHost));

    // Free the d_grid pointer
    CUDA_CALL(cudaFree(*d_grid));

    // Free the XN_flat, YN_flat, ZN_flat pointers
    CUDA_CALL(cudaFree(h_grid->XN_flat));
    CUDA_CALL(cudaFree(h_grid->YN_flat));
    CUDA_CALL(cudaFree(h_grid->ZN_flat));
}

void cudaMemcpyEMfieldHostToDevice(EMfield **const d_EMfield, const EMfield *const h_EMfield, const grid *const h_grid)
{
    /*
    SPECS similar to cudaMemcpyGridHostToDevice
    */
    // Shallow Copy a h_EMfield
    EMfield copyEMField = *h_EMfield;
    EMfield *const h_EMfield_copy = &copyEMField;

    // Define the arrays needed to be copied to device
    FPfield *d_Ex_flat;
    FPfield *d_Ey_flat;
    FPfield *d_Ez_flat;
    FPfield *d_Bxn_flat;
    FPfield *d_Byn_flat;
    FPfield *d_Bzn_flat;

    // Allocate the memory for these devices
    size_t arraySize = sizeof(FPfield) * h_grid->nxn * h_grid->nyn * h_grid->nzn;
    CUDA_CALL(cudaMalloc((void **)d_EMfield, sizeof(EMfield)));
    CUDA_CALL(cudaMalloc((void **)&d_Ex_flat, arraySize));
    CUDA_CALL(cudaMalloc((void **)&d_Ey_flat, arraySize));
    CUDA_CALL(cudaMalloc((void **)&d_Ez_flat, arraySize));
    CUDA_CALL(cudaMalloc((void **)&d_Bxn_flat, arraySize));
    CUDA_CALL(cudaMalloc((void **)&d_Byn_flat, arraySize));
    CUDA_CALL(cudaMalloc((void **)&d_Bzn_flat, arraySize));

    // Copy the arrays to the device
    CUDA_CALL(cudaMemcpy(d_Ex_flat, h_EMfield_copy->Ex_flat, arraySize, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_Ey_flat, h_EMfield_copy->Ey_flat, arraySize, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_Ez_flat, h_EMfield_copy->Ez_flat, arraySize, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_Bxn_flat, h_EMfield_copy->Bxn_flat, arraySize, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_Byn_flat, h_EMfield_copy->Byn_flat, arraySize, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_Bzn_flat, h_EMfield_copy->Bzn_flat, arraySize, cudaMemcpyHostToDevice));

    // Redirect pointers in h_EMfield_copy to devices
    h_EMfield_copy->Ex_flat = d_Ex_flat;
    h_EMfield_copy->Ey_flat = d_Ey_flat;
    h_EMfield_copy->Ez_flat = d_Ez_flat;
    h_EMfield_copy->Bxn_flat = d_Bxn_flat;
    h_EMfield_copy->Byn_flat = d_Byn_flat;
    h_EMfield_copy->Bzn_flat = d_Bzn_flat;

    // Copy the h_EMfield_copy struct to the device
    CUDA_CALL(cudaMemcpy(*d_EMfield, h_EMfield_copy, sizeof(EMfield), cudaMemcpyHostToDevice));
}

void cudaFreeEMfield(EMfield **const d_EMfield)
{
    /*
    SPECS similar to cudaFreeGridDeviceToHost
    */

    // Save a copy of h_EMfield, all pointers in there are allocated to host memory
    EMfield h_EMfield_instance;
    EMfield *h_EMfield = &h_EMfield_instance;

    // Copy the d_EMfield from device to host
    CUDA_CALL(cudaMemcpy(h_EMfield, *d_EMfield, sizeof(EMfield), cudaMemcpyDeviceToHost));

    // Free the d_EMfield pointer
    CUDA_CALL(cudaFree(*d_EMfield));

    // Free the Ex_flat, Ey_flat, Ez_flat, Bxn_flat, Byn_flat, Bzn_flat pointers
    CUDA_CALL(cudaFree(h_EMfield->Ex_flat));
    CUDA_CALL(cudaFree(h_EMfield->Ey_flat));
    CUDA_CALL(cudaFree(h_EMfield->Ez_flat));
    CUDA_CALL(cudaFree(h_EMfield->Bxn_flat));
    CUDA_CALL(cudaFree(h_EMfield->Byn_flat));
    CUDA_CALL(cudaFree(h_EMfield->Bzn_flat));
}

void cudaShallowMemcpyParamHostToDevice(parameters **const d_param, const parameters *const h_param)
{
    /*
    INPUT SPECIFICATIONS:
    d_param: a pointer to a param instance that is NOT allocated on the host or the device.
    h_param: a pointer to a param instance that should have all variable fields pre-allocated on the host with malloc or new.
    MODIFICATIONS:
    d_param: The param instance it is pointing to will be allocated memory to become a shallow copy of h_param. The pointers in d_param will be UNDEFINED.
    FUNCTIONALITY:
    Shallow Copy the h_param to device as d_param, while allocating memory for d_param.
    */
    CUDA_CALL(cudaMalloc((void **)d_param, sizeof(parameters)));
    CUDA_CALL(cudaMemcpy(*d_param, h_param, sizeof(parameters), cudaMemcpyHostToDevice));
}

void cudaShallowFreeParam(parameters **const d_param)
{
    /*
    INPUT SPECIFICATIONS:
    d_param: a pointer to a param instance that is allocated on the device, the pointers in d_param are UNDEFINED.
    MODIFICATIONS:
    d_param: Will be shallow cleaned.
    FUNCTIONALITY:
    Shallow clean d_param.
    */
    CUDA_CALL(cudaFree(*d_param));
}

/** allocate particle arrays */
void particle_allocate(struct parameters *param, struct particles *part, int is)
{

    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];

    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0)
    { // electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    }
    else
    { // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }

    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx * part->npcely * part->npcelz;

    // cast it to required precision
    part->qom = (FPpart)param->qom[is];

    long npmax = part->npmax;

    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart)param->u0[is];
    part->v0 = (FPpart)param->v0[is];
    part->w0 = (FPpart)param->w0[is];
    // thermal
    part->uth = (FPpart)param->uth[is];
    part->vth = (FPpart)param->vth[is];
    part->wth = (FPpart)param->wth[is];

    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
}
/** deallocate */
void particle_deallocate(struct particles *part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

/** particle mover */
int mover_PC(struct particles *part, struct EMfield *field, struct grid *grd, struct parameters *param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING " << param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart)param->dt / ((double)part->n_sub_cycles);
    FPpart dto2 = .5 * dt_sub_cycling, qomdt2 = part->qom * dto2 / param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;

    // local (to the particle) electric and magnetic field
    FPfield Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

    // interpolation densities
    int ix, iy, iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];

    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    // start subcycling
    for (int i_sub = 0; i_sub < part->n_sub_cycles; i_sub++)
    {
        // move each particle with new fields
        for (int i = 0; i < part->nop; i++)
        {
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for (int innter = 0; innter < part->NiterMover; innter++)
            {
                // interpolation G-->P
                ix = 2 + int((part->x[i] - grd->xStart) * grd->invdx);
                iy = 2 + int((part->y[i] - grd->yStart) * grd->invdy);
                iz = 2 + int((part->z[i] - grd->zStart) * grd->invdz);

                // calculate weights
                xi[0] = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0] = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1] = grd->XN[ix][iy][iz] - part->x[i];
                eta[1] = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;

                // set to zero local electric and magnetic field
                Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                        {
                            Exl += weight[ii][jj][kk] * field->Ex[ix - ii][iy - jj][iz - kk];
                            Eyl += weight[ii][jj][kk] * field->Ey[ix - ii][iy - jj][iz - kk];
                            Ezl += weight[ii][jj][kk] * field->Ez[ix - ii][iy - jj][iz - kk];
                            Bxl += weight[ii][jj][kk] * field->Bxn[ix - ii][iy - jj][iz - kk];
                            Byl += weight[ii][jj][kk] * field->Byn[ix - ii][iy - jj][iz - kk];
                            Bzl += weight[ii][jj][kk] * field->Bzn[ix - ii][iy - jj][iz - kk];
                        }

                // end interpolation
                omdtsq = qomdt2 * qomdt2 * (Bxl * Bxl + Byl * Byl + Bzl * Bzl);
                denom = 1.0 / (1.0 + omdtsq);
                // solve the position equation
                ut = part->u[i] + qomdt2 * Exl;
                vt = part->v[i] + qomdt2 * Eyl;
                wt = part->w[i] + qomdt2 * Ezl;
                udotb = ut * Bxl + vt * Byl + wt * Bzl;
                // solve the velocity equation
                uptilde = (ut + qomdt2 * (vt * Bzl - wt * Byl + qomdt2 * udotb * Bxl)) * denom;
                vptilde = (vt + qomdt2 * (wt * Bxl - ut * Bzl + qomdt2 * udotb * Byl)) * denom;
                wptilde = (wt + qomdt2 * (ut * Byl - vt * Bxl + qomdt2 * udotb * Bzl)) * denom;
                // update position
                part->x[i] = xptilde + uptilde * dto2;
                part->y[i] = yptilde + vptilde * dto2;
                part->z[i] = zptilde + wptilde * dto2;

            } // end of iteration
            // update the final position and velocity
            part->u[i] = 2.0 * uptilde - part->u[i];
            part->v[i] = 2.0 * vptilde - part->v[i];
            part->w[i] = 2.0 * wptilde - part->w[i];
            part->x[i] = xptilde + uptilde * dt_sub_cycling;
            part->y[i] = yptilde + vptilde * dt_sub_cycling;
            part->z[i] = zptilde + wptilde * dt_sub_cycling;

            //////////
            //////////
            ////////// BC

            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx)
            {
                if (param->PERIODICX == true)
                { // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                }
                else
                { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2 * grd->Lx - part->x[i];
                }
            }

            if (part->x[i] < 0)
            {
                if (param->PERIODICX == true)
                { // PERIODIC
                    part->x[i] = part->x[i] + grd->Lx;
                }
                else
                { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }

            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly)
            {
                if (param->PERIODICY == true)
                { // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                }
                else
                { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2 * grd->Ly - part->y[i];
                }
            }

            if (part->y[i] < 0)
            {
                if (param->PERIODICY == true)
                { // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                }
                else
                { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }

            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz)
            {
                if (param->PERIODICZ == true)
                { // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                }
                else
                { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2 * grd->Lz - part->z[i];
                }
            }

            if (part->z[i] < 0)
            {
                if (param->PERIODICZ == true)
                { // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                }
                else
                { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }

        } // end of subcycling
    }     // end of one particle

    return (0); // exit succcesfully
} // end of the mover

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles *part, struct interpDensSpecies *ids, struct grid *grd)
{

    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];

    // index of the cell
    int ix, iy, iz;

    for (register long long i = 0; i < part->nop; i++)
    {

        // determine cell: can we change to int()? is it faster?
        ix = 2 + int(floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int(floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int(floor((part->z[i] - grd->zStart) * grd->invdz));

        // distances from node
        xi[0] = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0] = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1] = grd->XN[ix][iy][iz] - part->x[i];
        eta[1] = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];

        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;

        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;

        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];

        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pzz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    }
}

__global__ void mover_PC_gpu_kernel(struct particles *part, struct EMfield *field, struct grid *grd, struct parameters *param)
{
    const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= part->nop)
        return;

    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart)param->dt / ((double)part->n_sub_cycles);
    FPpart dto2 = .5 * dt_sub_cycling, qomdt2 = part->qom * dto2 / param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;

    // local (to the particle) electric and magnetic field
    FPfield Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

    // interpolation densities
    int ix, iy, iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];

    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    for (int i_sub = 0; i_sub < part->n_sub_cycles; i_sub++)
    {
        xptilde = part->x[i];
        yptilde = part->y[i];
        zptilde = part->z[i];
        // calculate the average velocity iteratively
        for (int innter = 0; innter < part->NiterMover; innter++)
        {
            // interpolation G-->P
            ix = 2 + int((part->x[i] - grd->xStart) * grd->invdx);
            iy = 2 + int((part->y[i] - grd->yStart) * grd->invdy);
            iz = 2 + int((part->z[i] - grd->zStart) * grd->invdz);

            // calculate weights

            xi[0] = part->x[i] - grd->XN_flat[get_idx(ix - 1, iy, iz, grd->nyn, grd->nzn)];
            eta[0] = part->y[i] - grd->YN_flat[get_idx(ix, iy - 1, iz, grd->nyn, grd->nzn)];
            zeta[0] = part->z[i] - grd->ZN_flat[get_idx(ix, iy, iz - 1, grd->nyn, grd->nzn)];
            xi[1] = grd->XN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part->x[i];
            eta[1] = grd->YN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part->y[i];
            zeta[1] = grd->ZN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part->z[i];
            for (int ii = 0; ii < 2; ii++)
                for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++)
                        weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;

            // set to zero local electric and magnetic field
            Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

            for (int ii = 0; ii < 2; ii++)
                for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++)
                    {
                        Exl += weight[ii][jj][kk] * field->Ex_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                        Eyl += weight[ii][jj][kk] * field->Ey_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                        Ezl += weight[ii][jj][kk] * field->Ez_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                        Bxl += weight[ii][jj][kk] * field->Bxn_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                        Byl += weight[ii][jj][kk] * field->Byn_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                        Bzl += weight[ii][jj][kk] * field->Bzn_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                    }

            // end interpolation
            omdtsq = qomdt2 * qomdt2 * (Bxl * Bxl + Byl * Byl + Bzl * Bzl);
            denom = 1.0 / (1.0 + omdtsq);
            // solve the position equation
            ut = part->u[i] + qomdt2 * Exl;
            vt = part->v[i] + qomdt2 * Eyl;
            wt = part->w[i] + qomdt2 * Ezl;
            udotb = ut * Bxl + vt * Byl + wt * Bzl;
            // solve the velocity equation
            uptilde = (ut + qomdt2 * (vt * Bzl - wt * Byl + qomdt2 * udotb * Bxl)) * denom;
            vptilde = (vt + qomdt2 * (wt * Bxl - ut * Bzl + qomdt2 * udotb * Byl)) * denom;
            wptilde = (wt + qomdt2 * (ut * Byl - vt * Bxl + qomdt2 * udotb * Bzl)) * denom;
            // update position
            part->x[i] = xptilde + uptilde * dto2;
            part->y[i] = yptilde + vptilde * dto2;
            part->z[i] = zptilde + wptilde * dto2;

        } // end of iteration
        // update the final position and velocity
        part->u[i] = 2.0 * uptilde - part->u[i];
        part->v[i] = 2.0 * vptilde - part->v[i];
        part->w[i] = 2.0 * wptilde - part->w[i];
        part->x[i] = xptilde + uptilde * dt_sub_cycling;
        part->y[i] = yptilde + vptilde * dt_sub_cycling;
        part->z[i] = zptilde + wptilde * dt_sub_cycling;

        //////////
        //////////
        ////////// BC

        // X-DIRECTION: BC particles
        if (part->x[i] > grd->Lx)
        {
            if (param->PERIODICX == true)
            { // PERIODIC
                part->x[i] = part->x[i] - grd->Lx;
            }
            else
            { // REFLECTING BC
                part->u[i] = -part->u[i];
                part->x[i] = 2 * grd->Lx - part->x[i];
            }
        }

        if (part->x[i] < 0)
        {
            if (param->PERIODICX == true)
            { // PERIODIC
                part->x[i] = part->x[i] + grd->Lx;
            }
            else
            { // REFLECTING BC
                part->u[i] = -part->u[i];
                part->x[i] = -part->x[i];
            }
        }

        // Y-DIRECTION: BC particles
        if (part->y[i] > grd->Ly)
        {
            if (param->PERIODICY == true)
            { // PERIODIC
                part->y[i] = part->y[i] - grd->Ly;
            }
            else
            { // REFLECTING BC
                part->v[i] = -part->v[i];
                part->y[i] = 2 * grd->Ly - part->y[i];
            }
        }

        if (part->y[i] < 0)
        {
            if (param->PERIODICY == true)
            { // PERIODIC
                part->y[i] = part->y[i] + grd->Ly;
            }
            else
            { // REFLECTING BC
                part->v[i] = -part->v[i];
                part->y[i] = -part->y[i];
            }
        }

        // Z-DIRECTION: BC particles
        if (part->z[i] > grd->Lz)
        {
            if (param->PERIODICZ == true)
            { // PERIODIC
                part->z[i] = part->z[i] - grd->Lz;
            }
            else
            { // REFLECTING BC
                part->w[i] = -part->w[i];
                part->z[i] = 2 * grd->Lz - part->z[i];
            }
        }

        if (part->z[i] < 0)
        {
            if (param->PERIODICZ == true)
            { // PERIODIC
                part->z[i] = part->z[i] + grd->Lz;
            }
            else
            { // REFLECTING BC
                part->w[i] = -part->w[i];
                part->z[i] = -part->z[i];
            }
        }
    } // end of subcycling
} // end of mover gpu kernel

/** particle mover */
int mover_PC_gpu(struct particles *part, struct EMfield *field, struct grid *grd, struct parameters *param)
{
    std::cout << "***  MOVER with SUBCYCLYING " << param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
    particles *d_part;
    EMfield *d_field;
    grid *d_grd;
    parameters *d_param;

    cudaMemcpyParticlesHostToDevice(&d_part, part);
    cudaMemcpyEMfieldHostToDevice(&d_field, field, grd);
    cudaMemcpyGridHostToDevice(&d_grd, grd);
    cudaShallowMemcpyParamHostToDevice(&d_param, param);

    mover_PC_gpu_kernel<<<((part->nop / THREAD_PER_BLOCK) + 1), THREAD_PER_BLOCK>>>(d_part, d_field, d_grd, d_param);
    cudaDeviceSynchronize();

    cudaMemcpyParticlesDeviceToHost(part, &d_part);
    cudaFreeEMfield(&d_field);
    cudaFreeGrid(&d_grd);
    cudaShallowFreeParam(&d_param);

    return 0;
} // end of the mover