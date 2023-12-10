#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define TPB 32

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
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
void particle_deallocate(struct particles* part)
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

__global__ void mover_PC_kernel(FPpart* part_x  , FPpart* part_y  , FPpart* part_z  ,
                                FPpart* part_u  , FPpart* part_v  , FPpart* part_w  ,
                                FPfield* fieldEx , FPfield* fieldEy , FPfield* fieldEz ,
                                FPfield* fieldBxn, FPfield* fieldByn, FPfield* fieldBzn,
                                FPfield* grd_XN , FPfield* grd_YN , FPfield* grd_ZN ,
                                int nop   , int n_sub_cycles, int NiterMover, FPpart dt_sub_cycling, 
                                FPpart dto2, FPpart qomdt2, struct grid grd, struct parameters param)
{
    // thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nop) return;

    // auxiliary variables
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    xptilde = part_x[i];
    yptilde = part_y[i];
    zptilde = part_z[i];
    
    // start subcycling
    for (int i_sub=0; i_sub <  n_sub_cycles; i_sub++){
            // calculate the average velocity iteratively
            for(int innter=0; innter < NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part_x[i] - grd.xStart)*grd.invdx);
                iy = 2 +  int((part_y[i] - grd.yStart)*grd.invdy);
                iz = 2 +  int((part_z[i] - grd.zStart)*grd.invdz);
                
                // calculate weights
                xi[0]   = part_x[i] - grd_XN[get_idx(ix - 1, iy, iz)];
                eta[0]  = part_y[i] - grd_YN[get_idx(ix, iy - 1, iz)];
                zeta[0] = part_z[i] - grd_ZN[get_idx(ix, iy, iz - 1)];
                xi[1]   = grd_XN[get_idx(ix, iy, iz)] - part_x[i];
                eta[1]  = grd_YN[get_idx(ix, iy, iz)] - part_y[i];
                zeta[1] = grd_ZN[get_idx(ix, iy, iz)] - part_z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd.invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*fieldEx[get_idx(ix- ii, iy -jj, iz- kk)];
                            Eyl += weight[ii][jj][kk]*fieldEy[get_idx(ix- ii, iy -jj, iz- kk)];
                            Ezl += weight[ii][jj][kk]*fieldEz[get_idx(ix- ii, iy -jj, iz- kk)];
                            Bxl += weight[ii][jj][kk]*fieldBxn[get_idx(ix- ii, iy -jj, iz- kk)];
                            Byl += weight[ii][jj][kk]*fieldByn[get_idx(ix- ii, iy -jj, iz- kk)];
                            Bzl += weight[ii][jj][kk]*fieldBzn[get_idx(ix- ii, iy -jj, iz- kk)];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part_u[i] + qomdt2*Exl;
                vt= part_v[i] + qomdt2*Eyl;
                wt= part_w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part_x[i] = xptilde + uptilde*dto2;
                part_y[i] = yptilde + vptilde*dto2;
                part_z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part_u[i]= 2.0*uptilde - part_u[i];
            part_v[i]= 2.0*vptilde - part_v[i];
            part_w[i]= 2.0*wptilde - part_w[i];
            part_x[i] = xptilde + uptilde*dt_sub_cycling;
            part_y[i] = yptilde + vptilde*dt_sub_cycling;
            part_v[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part_x[i] > grd.Lx){
                if (param.PERIODICX==true){ // PERIODIC
                    part_x[i] = part_x[i] - grd.Lx;
                } else { // REFLECTING BC
                    part_u[i] = -part_u[i];
                    part_x[i] = 2*grd.Lx - part_x[i];
                }
            }
                                                                        
            if (part_x[i] < 0){
                if (param.PERIODICX==true){ // PERIODIC
                   part_x[i] = part_x[i] + grd.Lx;
                } else { // REFLECTING BC
                    part_u[i] = -part_u[i];
                    part_x[i] = -part_x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part_y[i] > grd.Ly){
                if (param.PERIODICY==true){ // PERIODIC
                    part_y[i] = part_y[i] - grd.Ly;
                } else { // REFLECTING BC
                    part_v[i] = -part_v[i];
                    part_y[i] = 2*grd.Ly - part_y[i];
                }
            }
                                                                        
            if (part_y[i] < 0){
                if (param.PERIODICY==true){ // PERIODIC
                    part_y[i] = part_v[i] + grd.Ly;
                } else { // REFLECTING BC
                    part_v[i] = -part_v[i];
                    part_y[i] = -part_y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part_z[i] > grd.Lz){
                if (param.PERIODICZ==true){ // PERIODIC
                    part_z[i] = part_z[i] - grd.Lz;
                } else { // REFLECTING BC
                    part_w[i] = -part_w[i];
                    part_z[i] = 2*grd.Lz - part_z[i];
                }
            }
                                                                        
            if (part_z[i] < 0){
                if (param.PERIODICZ==true){ // PERIODIC
                    part_z[i] = part_z[i] + grd.Lz;
                } else { // REFLECTING BC
                    part_w[i] = -part_w[i];
                    part_z[i] = -part_z[i];
                }
            }
                                                                        
    }
}


/** GPU particle mover */
int mover_PC_gpu(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;

    //////////////////////////////////////////////////////////////////////////////////
    FPpart *part_x; FPpart *part_y; FPpart *part_z;
    FPpart *part_u; FPpart *part_v; FPpart *part_w;
    FPfield *grd_XN; FPfield *grd_YN; FPfield *grd_ZN;
    FPfield *fieldEx ; FPfield *fieldEy ; FPfield *fieldEz;
    FPfield *fieldBxn; FPfield *fieldByn; FPfield *fieldBzn;

    struct grid grr = *grd;
    long field_size = grr.nxn * grr.nyn * grr.nzn;
    long grd_size   = grr.nxn * grr.nyn * grr.nzn;
    int nop = part->nop;

    cudaMalloc(&part_x, part->npmax * sizeof(FPpart));
    cudaMalloc(&part_y, part->npmax * sizeof(FPpart));
    cudaMalloc(&part_z, part->npmax * sizeof(FPpart));
    cudaMalloc(&part_u, part->npmax * sizeof(FPpart));
    cudaMalloc(&part_v, part->npmax * sizeof(FPpart));
    cudaMalloc(&part_w, part->npmax * sizeof(FPpart));
    cudaMalloc(&fieldEx , field_size * sizeof(FPfield));
    cudaMalloc(&fieldEy , field_size * sizeof(FPfield));
    cudaMalloc(&fieldEz , field_size * sizeof(FPfield));
    cudaMalloc(&fieldBxn, field_size * sizeof(FPfield));
    cudaMalloc(&fieldByn, field_size * sizeof(FPfield));
    cudaMalloc(&fieldBzn, field_size * sizeof(FPfield));
    cudaMalloc(&grd_XN, grd_size * sizeof(FPfield));    
    cudaMalloc(&grd_YN, grd_size * sizeof(FPfield));
    cudaMalloc(&grd_ZN, grd_size * sizeof(FPfield));
        
    cudaMemcpy(part_x, part->x, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(part_y, part->y, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(part_z, part->z, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(part_u, part->u, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(part_v, part->v, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(part_w, part->w, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(fieldEx, field->Ex_flat, field_size * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(fieldEy, field->Ey_flat, field_size * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(fieldEz, field->Ez_flat, field_size * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(fieldBxn, field->Bxn_flat, field_size * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(fieldByn, field->Byn_flat, field_size * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(fieldBzn, field->Bzn_flat, field_size * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(grd_XN, grd->XN_flat, grd_size * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(grd_YN, grd->YN_flat, grd_size * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(grd_ZN, grd->ZN_flat, grd_size * sizeof(FPfield), cudaMemcpyHostToDevice);
    
    mover_PC_kernel<<<(nop+TPB-1)/TPB, TPB>>>(part_x, part_y, part_z,
                                            part_u, part_v, part_w,
                                            fieldEx, fieldEy, fieldEz,
                                            fieldBxn, fieldByn, fieldBzn,
                                            grd_XN, grd_YN, grd_ZN,
                                            part->nop, part->n_sub_cycles, part->NiterMover,
                                            dt_sub_cycling, dto2, qomdt2, *grd, *param);

    cudaMemcpy(part->x, part_x, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->y, part_y, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->z, part_z, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->u, part_u, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->v, part_v, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->w, part_w, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Ex_flat, fieldEx, field_size * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Ey_flat, fieldEy, field_size * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Ez_flat, fieldEz, field_size * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Bxn_flat, fieldBxn, field_size * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Byn_flat, fieldByn, field_size * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(field->Bzn_flat, fieldBzn, field_size * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(grd->XN_flat, grd_XN, grd_size * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(grd->YN_flat, grd_YN, grd_size * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(grd->ZN_flat, grd_ZN, grd_size * sizeof(FPfield), cudaMemcpyDeviceToHost);

    cudaFree(part_x);
    cudaFree(part_y);
    cudaFree(part_z);
    cudaFree(part_u);
    cudaFree(part_v);
    cudaFree(part_w);
    cudaFree(fieldEx);
    cudaFree(fieldEy);
    cudaFree(fieldEz);
    cudaFree(fieldBxn);
    cudaFree(fieldByn);
    cudaFree(fieldBzn);
    cudaFree(grd_XN);
    cudaFree(grd_YN);
    cudaFree(grd_ZN);
                                                                        
    return(0);
}


/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i <  part->nop; i++){
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }
                                                                        
            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }
                                                                        
            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }
                                                                        
            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
                                                                        
            
            
        }  // end of subcycling
    } // end of one particle
                                                                        
    return(0); // exit succcesfully
} // end of the mover



/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
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
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}
