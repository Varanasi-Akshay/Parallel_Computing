#include <stdlib.h>
#include <stdio.h>
#include <petsc.h>

PetscErrorCode MatTridiagonal(Mat A) {
  PetscErrorCode ierr;
  int N;
  PetscFunctionBegin;
  int first,last;
  ierr = MatGetOwnershipRange(A,&first,&last); CHKERRQ(ierr);
  ierr = MatGetSize(A,&N,PETSC_NULL); CHKERRQ(ierr);
  for (int i=first; i<last; i++) {
    ierr = MatSetValue(A,i,i,2.,INSERT_VALUES); CHKERRQ(ierr);
    if (i>0) {
      ierr = MatSetValue(A,i,i-1,2.,INSERT_VALUES); CHKERRQ(ierr);
    }
    if (i<N-1) {
      ierr = MatSetValue(A,i,i+1,2.,INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv) {
  MPI_Comm comm;
  PetscErrorCode ierr;
  
  PetscInitialize(&argc,&argv,0,0);
  comm = PETSC_COMM_WORLD;
  
  int N = 30000;
  ierr = PetscOptionsGetInt
    (PETSC_NULL,
#if PETSC_VERSION_MINOR > 6
     PETSC_NULL,
#endif
     "-nlocal",&N,PETSC_NULL); CHKERRQ(ierr);
  Mat A;
  ierr = MatCreate(comm,&A); CHKERRQ(ierr);
  ierr = MatSetType(A,MATMPIAIJ); CHKERRQ(ierr);
  ierr = MatSetSizes(A,N,N,PETSC_DECIDE,PETSC_DECIDE); CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,3,PETSC_NULL,3,PETSC_NULL); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FLUSH_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FLUSH_ASSEMBLY); CHKERRQ(ierr);

  ierr = MatTridiagonal(A); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  KSP solver;
  ierr = KSPCreate(comm,&solver); CHKERRQ(ierr);
  ierr = KSPSetType(solver,KSPCG); CHKERRQ(ierr);
  {
    PC precon;
    ierr = KSPGetPC(solver,&precon); CHKERRQ(ierr);
    ierr = PCSetType(precon,PCJACOBI); CHKERRQ(ierr);
  }
  ierr = KSPSetOperators(solver,A,A); CHKERRQ(ierr);

  Vec x,y;
  ierr = VecCreate(comm,&x); CHKERRQ(ierr);
  ierr = VecSetType(x,VECMPI); CHKERRQ(ierr);
  ierr = VecSetSizes(x,N,PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(x); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x); CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y); CHKERRQ(ierr);

  ierr = KSPSetNormType(solver,KSP_NORM_NATURAL); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(solver); CHKERRQ(ierr);
  ierr = KSPSetUp(solver); CHKERRQ(ierr);
  ierr = KSPSolve(solver,x,y); CHKERRQ(ierr);

  PetscFinalize();
  return 0;
}
