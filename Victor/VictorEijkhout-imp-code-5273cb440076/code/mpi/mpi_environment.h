// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014/5
 ****
 **** mpi_environment.h: headers for defining an MPI environment
 ****
 ****************************************************************/

#ifndef MPI_ENV_H
#define MPI_ENV_H

#include "imp_base.h"


class mpi_architecture;
class mpi_environment : public environment {
public: // data
  MPI_Comm comm;
  int processor_grouping{0}; // let's take stampede as default
public: // methods
  mpi_environment(int argc,char **argv);
  ~mpi_environment();
  mpi_environment( mpi_environment& other ) : environment( other ) {
    comm = other.comm; };
  void mpi_delete_environment(); // extra deletions after the destructor
  //  virtual void delete_environment() override;

  void set_mpi_environment();
  virtual architecture *make_architecture();
  virtual void get_comm(void *ptr) override { *(MPI_Comm*)ptr = comm; };
  int get_processor_grouping() { return processor_grouping; };
  void set_processor_grouping(int g) { processor_grouping = g; };
  virtual void print_options() override;
  virtual int iargument(const char *argname,int vdef) override {
    int val,mytid; MPI_Comm_rank(comm,&mytid);
    if (mytid==0) {
      //      val = static_cast<environment*>(this)->iargument(argname,vdef);
      val = environment::iargument(argname,vdef);
    }
    MPI_Bcast(&val,1,MPI_INT,0,comm);
    return val;
  };
  result_tuple *mpi_summarize_entities();
  virtual void print_all(std::string s) override;
  virtual void tasks_to_dot_file() override;
};

#endif
