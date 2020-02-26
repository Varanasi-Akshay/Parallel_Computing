// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** product_ops.h: Header file for the MPI+OMP operation kernels
 ****
 ****************************************************************/

#ifndef PRODUCT_OPS_H
#define PRODUCT_OPS_H 1

#include "product_base.h"
#include "imp_functions.h"
#include "mpi_ops.h"
#include "imp_ops.h"

/*!
  Copying is simple
*/
class product_copy_kernel : public product_kernel,public copy_kernel {
public:
  product_copy_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),copy_kernel(in,out),product_kernel(in,out),
      entity(entity_cookie::KERNEL) {};
};

/*!
  Scale a vector by a scalar.
*/
class product_scale_kernel : public product_kernel,public scale_kernel {
public:
  product_scale_kernel( double *a,std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),scale_kernel(a,in,out),product_kernel(in,out),
      entity(entity_cookie::KERNEL)  {};
  product_scale_kernel( std::shared_ptr<object> a,std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),scale_kernel(a,in,out),product_kernel(in,out),
      entity(entity_cookie::KERNEL)  {};
};

/*!
  Scale a vector down by a scalar.
*/
class product_scaledown_kernel : public product_kernel,public scaledown_kernel {
public:
  product_scaledown_kernel( double *a,std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),scaledown_kernel(a,in,out),product_kernel(in,out),
      entity(entity_cookie::KERNEL)  {};
  product_scaledown_kernel( std::shared_ptr<object> a,std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),scaledown_kernel(a,in,out),product_kernel(in,out),
      entity(entity_cookie::KERNEL)  {};
};

/*!
  AXPY is a local kernel; the scalar is passed as the context. The scalar
  comes in as double*, this leaves open the possibility of an array of scalars,
  also it puts my mind at ease re that casting to void*.
*/
class product_axpy_kernel : public product_kernel,public axpy_kernel {
public:
  product_axpy_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out,double *x )
    : kernel(in,out),axpy_kernel(in,out,x),product_kernel(in,out),
      entity(entity_cookie::KERNEL)  {};
};

/*!
  Add two vectors together. For a more general version see \ref product_axbyz_kernel.
 */
class product_sum_kernel : public product_kernel,public sum_kernel {
public:
  product_sum_kernel( std::shared_ptr<object> in1,std::shared_ptr<object> in2,std::shared_ptr<object> out )
    : kernel(in1,out),sum_kernel(in1,in2,out),product_kernel(in1,out),
      entity(entity_cookie::KERNEL)  {};
};

/*!
  Product version of the \ref axbyz_kernel.
*/
class product_axbyz_kernel : public product_kernel,public axbyz_kernel {
protected:
public:
  product_axbyz_kernel( char op1,std::shared_ptr<object> s1,std::shared_ptr<object> x1,
		    char op2,std::shared_ptr<object> s2,std::shared_ptr<object> x2,std::shared_ptr<object> out )
    : kernel(x1,out),product_kernel(x1,out),axbyz_kernel(op1,s1,x1,op2,s2,x2,out),
      entity(entity_cookie::KERNEL) {};
  //! Abbreviated creation leaving the scalars untouched
  product_axbyz_kernel( std::shared_ptr<object> s1,std::shared_ptr<object> x1, std::shared_ptr<object> s2,std::shared_ptr<object> x2,std::shared_ptr<object> out ) :
    product_axbyz_kernel( '+',s1,x1, '+',s2,x2, out ) {}
};

// /*!
//   AXBYZ is a local kernel; the scalar is passed as the context. The scalar
//   comes in as double*, this leaves open the possibility of an array of scalars,
//   also it puts my mind at ease re that casting to void*.
// */
// class product_axbyz_kernel : virtual public product_kernel {
// protected:
// public:
//   product_axbyz_kernel( char op1,std::shared_ptr<object> s1,std::shared_ptr<object> x1,
// 		    char op2,std::shared_ptr<object> s2,std::shared_ptr<object> x2,std::shared_ptr<object> out )
//     : kernel(x1,out),product_kernel(x1,out) {
//     if (s1->get_distribution()->get_type()!=distribution_type::REPLICATED)
//       throw("s1 not replicated\n");
//     if (s2->get_distribution()->get_type()!=distribution_type::REPLICATED)
//       throw("s2 not replicated\n");
//     dependency *d;
//     set_name("ax+by=z");

//     // x1 is local
//     d = last_dependency(); d->set_name("wait x1"); d->set_type_local();
//     // s1 is local
//     add_in_object(s1);
//     d = last_dependency(); d->set_name("wait s1"); d->set_explicit_beta_distribution(s1);    
//     // x2 is local
//     add_in_object(x2);
//     d = last_dependency(); d->set_name("wait x2"); d->set_type_local();
//     // s2 is local
//     add_in_object(s2);
//     d = last_dependency(); d->set_name("wait s2"); d->set_explicit_beta_distribution(s2);

//     // function and context   
//     set_localexecutefn( &vecaxbyz );
//     charcharxyz_object_struct *ctx = new charcharxyz_object_struct;
//     ctx->c1 = op1; ctx->s1 = s1;
//     ctx->c2 = op2; ctx->s2 = s2; ctx->obj = x2;
//     set_localexecutectx( ctx );
// #ifdef VT
//     VT_funcdef("axbyz kernel",VT_NOCLASS,&vt_kernel_class);
// #endif
//   };
//   //! Abbreviated creation leaving the scalars untouched
//   product_axbyz_kernel( std::shared_ptr<object> s1,std::shared_ptr<object> x1, std::shared_ptr<object> s2,std::shared_ptr<object> x2,std::shared_ptr<object> out ) :
//     product_axbyz_kernel( '+',s1,x1, '+',s2,x2, out ) {};
// };

class product_norm_kernel : public product_kernel,public norm_kernel {
public:
  product_norm_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),product_kernel(in,out),norm_kernel(in,out),
      entity(entity_cookie::KERNEL) {};
};

//// VERY TEMPORARY !!!!
class product_reduction_kernel : virtual public product_kernel {
public:
  product_reduction_kernel( std::shared_ptr<object> local_value,std::shared_ptr<object> global_sum)
    : kernel(local_value,global_sum),product_kernel(local_value,global_sum),
      entity(entity_cookie::KERNEL) {};
};

#if 0

/*!
  A whole class for operations between replicated scalars
*/
class product_scalar_kernel : public product_kernel,public scalar_kernel {
protected:
public:
  product_scalar_kernel( std::shared_ptr<object> in1,const char *op,std::shared_ptr<object> in2,std::shared_ptr<object> out )
    : kernel(in1,out),scalar_kernel(in1,op,in2,out),product_kernel(in1,out),
      entity(entity_cookie::KERNEL) {};
};

/*! A sparse matrix-vector product is easily defined
  from the local product routine and using the 
  same sparse matrix as the index pattern of the beta distribution
*/
class product_spmvp_kernel : virtual public product_kernel {
public:
  product_spmvp_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out,product_sparse_matrix *mat)
    : kernel(in,out),product_kernel(in,out),
      entity(entity_cookie::KERNEL) {
    set_name("sparse-mvp");
    dependency *d = last_dependency();
    d->set_index_pattern( mat );
    { fmt::MemoryWriter w; w.write("spmvp-into-{}",out->get_name());
      d->set_name(w.c_str()); }
#ifdef VT
    VT_funcdef("spmvp kernel",VT_NOCLASS,&vt_kernel_class);
#endif
    localexecutefn = &local_sparse_matrix_vector_multiply;
    localexecutectx = (void*)mat;
  };
  //! We perform the regular kernel analysis, but also remap the matrix to the beta
  virtual void analyze_dependencies() override {
    product_kernel::analyze_dependencies();
    ( (product_sparse_matrix*)localexecutectx )->remap
      ( get_out_object()->get_distribution(),get_beta_distribution(0),get_architecture()->mytid() );
  };
};

//! \todo get that set_name to work.
class product_sidewaysdown_kernel : virtual public product_kernel {
private:
  distribution *level_dist,*half_dist;
  std::shared_ptr<object> expanded,multiplied;
  product_kernel *expand,*multiply,*sum;
public:
  product_sidewaysdown_kernel( std::shared_ptr<object> top,std::shared_ptr<object> side,std::shared_ptr<object> out,product_sparse_matrix *mat )
    : kernel(top,out),product_kernel(top,out),
      entity(entity_cookie::KERNEL) {
    level_dist = out->get_distribution(); half_dist = top->get_distribution();

    expanded = new product_object(level_dist);
    expand = new product_kernel(top,expanded);
    expand->set_localexecutefn( &scanexpand );
    expand->set_signature_function_function( &halfinterval );
    expand->set_name("sidewaysdown-expand");

    multiplied = new product_object(level_dist);
    multiply = new product_spmvp_kernel(side,multiplied,mat);
    multiply->set_name("sidewaysdown-multiply");
    
    sum = new product_sum_kernel(expanded,multiplied,out);
    sum->set_name("sidewaysdown-sum");
#ifdef VT
    VT_funcdef("sidewaysdown kernel",VT_NOCLASS,&vt_kernel_class);
#endif
  };
  ~product_sidewaysdown_kernel() { delete expanded; delete multiplied;
    delete expand; delete multiply; delete sum;
  };
  // void set_name( std::string s ) override {
  //   { fmt::MemoryWriter w; w.write("{}-expand",s);   expand->set_name(w.str()); }
  //   { fmt::MemoryWriter w; w.write("{}-multiply",s); multiply->set_name(w.str()); }
  //   { fmt::MemoryWriter w; w.write("{}-sum",s);      sum->set_name(w.str()); }
  // }
  /*! We override the default analysis by analyzing in sequence the contained kernels.
    This splits the kernels and analyzes task dependencies; then we take the tasks
    and add them to the sidewaysdown task list.
   */
  virtual void analyze_dependencies() override {
    std::vector<task*> *rtasks = new std::vector<task*>, *tmptasks;    

    expand->analyze_dependencies(); multiply->analyze_dependencies();
    sum->analyze_dependencies();

    tmptasks = expand->get_tasks();
    for (auto t=tmptasks->begin(); t!=tmptasks->end(); ++t)
      rtasks->push_back( *t );
    tmptasks = multiply->get_tasks();
    for (auto t=tmptasks->begin(); t!=tmptasks->end(); ++t)
      rtasks->push_back( *t );
    tmptasks = sum->get_tasks();
    for (auto t=tmptasks->begin(); t!=tmptasks->end(); ++t)
      rtasks->push_back( *t );
    this->set_kernel_tasks( rtasks );
  };
  std::string as_string() override {
    fmt::MemoryWriter w;
    w.write("K[{}]=\n",get_name());
    w.write("  <<{}>>\n",expand->as_string());
    w.write("  <<{}>>\n",multiply->as_string());
    w.write("  <<{}>>",sum->as_string());
    return w.str();
  };
};

class product_centerofmass_kernel : virtual public product_kernel {
public:
  product_centerofmass_kernel(std::shared_ptr<object> bot,std::shared_ptr<object> top,int k)
    : kernel(bot,top),product_kernel(bot,top),
      entity(entity_cookie::KERNEL) {
    set_localexecutefn( &scansum );
    set_localexecutectx( (void*)( new int[1]{k} ) );
    set_signature_function_function( &doubleinterval );
  };
};

/*!
  For now the reduction is only a sum reduction.

  \todo parametrize this with the combination function.
  \todo write a separate page about composite kernels?

  An reduction is somewhat tricky. While it behaves as a single kernel,
  it is composed of two kernels: one local sum followed by a gather,
  followed by a redundantly computed local sum.

  This means we have to overwrite #analyze_dependencies and #execute
  to call the corresponding routines of the three enclosed kernels.

  \todo test  that "global_sum" is replicated_scalar et cetera
  \todo make constructor s/t top_summing replicated_distribution( new indexstruct(0,ngroups) )
 */
class product_reduction_kernel : virtual public product_kernel {
private: // we need to keep them just to destroy them
  distribution *local_scalar,*gathered_scalar;
protected:
  product_distribution *locally_grouped{nullptr},*partially_reduced{nullptr};
  product_std::shared_ptr<object> partial_sums{nullptr};
  product_kernel *sumkernel,*groupkernel{nullptr};
  product_kernel *partial_summing,*top_summing;
public:
  product_reduction_kernel( std::shared_ptr<object> local_value,std::shared_ptr<object> global_sum)
    : kernel(local_value,global_sum),product_kernel(local_value,global_sum),
      entity(entity_cookie::KERNEL) {
    //    product_environment *env = (product_environment*)local_value->get_environment();
    architecture *arch = global_sum;
    set_name("scalar reduction");
    if (global_sum->has_collective_strategy(collective_strategy::GROUP)) {

      int
        P=get_architecture()->nprocs(), ntids=P, mytid=get_architecture()->mytid(), g=P-1;
      int groupsize;
      // if (env->get_processor_grouping()>0)
      //   groupsize = env->get_processor_grouping();
      // else
      groupsize = 4*( (sqrt(P)+3)/4 );

      //if (mytid==0) printf("grouping with %d\n",groupsize);
      int
        mygroupnum = mytid/groupsize, nfullgroups = ntids/groupsize,
        grouped_tids = nfullgroups*groupsize, // how many procs are in perfect groups?
        remainsize = P-grouped_tids, ngroups = nfullgroups+(remainsize>0);
      parallel_indexstruct *groups = new parallel_indexstruct(arch);
      for (int p=0; p<P; p++) {
        index_int groupnumber = p/groupsize,
          f = groupsize*groupnumber,l=MIN(f+groupsize-1,g);
        groups->set_processor_structure(p, new contiguous_indexstruct(f,l) );
      }
      locally_grouped = new product_distribution(arch,groups);

      parallel_indexstruct *partials = new parallel_indexstruct(arch);
      for (int p=0; p<P; p++) {
        index_int groupnumber = p/groupsize;
        partials->set_processor_structure(p, new contiguous_indexstruct(groupnumber) );
      }
      partially_reduced = new product_distribution(arch,partials);

      partial_sums = new product_object(partially_reduced);
      partial_summing = new product_kernel(local_value,partial_sums);
      partial_summing->set_name("reduction:partial-sum-local-to-group");
      partial_summing->set_explicit_beta_distribution(locally_grouped);
      partial_summing->set_localexecutefn( &summing );

      top_summing = new product_kernel(partial_sums,global_sum);
      top_summing->set_name("reduction:group-sum-to-global");
      parallel_indexstruct *top_beta = new parallel_indexstruct(arch);
      for (int p=0; p<P; p++)
        top_beta->set_processor_structure(p, new contiguous_indexstruct(0,ngroups-1));
      top_summing->set_explicit_beta_distribution
        ( new product_distribution(arch,top_beta) );
      top_summing->set_localexecutefn( &summing );
    } else if (global_sum->has_collective_strategy(collective_strategy::ALL_PTP)) {
      sumkernel = new product_kernel(local_value,global_sum);
      sumkernel->set_name("reduction:one-step-sum");
      gathered_scalar = new product_gathered_distribution(arch);
      sumkernel->set_explicit_beta_distribution( new product_gathered_distribution(arch) );
      sumkernel->set_localexecutefn( &summing );
    } else {printf("strategy %d\n",-1); //env->get_collective_strategy());
      throw("Unknown collective strategy\n");}
#ifdef VT
    VT_funcdef("reduction kernel",VT_NOCLASS,&vt_kernel_class);
#endif
  };
  // ~product_reduction() { delete local_scalar; delete gathered_scalar;
  //   delete local_value; delete prekernel; delete sumkernel; };
  //! We override the default analysis by analyzing in sequence the contained kernels.
  virtual void analyze_dependencies() override {
    architecture *arch = get_out_object();
    //product_environment *env = (product_environment*)get_out_object()->get_environment();
    std::vector<task*> *rtasks = new std::vector<task*>, *tmptasks;

    if (get_out_object()->has_collective_strategy(collective_strategy::GROUP)) {
      partial_summing->analyze_dependencies();
      tmptasks = partial_summing->get_tasks();
      for (std::vector<task*>::iterator t=tmptasks->begin(); t!=tmptasks->end(); ++t)
	rtasks->push_back( *t );

      top_summing->analyze_dependencies();
      tmptasks = top_summing->get_tasks();
      for (std::vector<task*>::iterator t=tmptasks->begin(); t!=tmptasks->end(); ++t)
	rtasks->push_back( *t );
    } else if (arch->has_collective_strategy(collective_strategy::ALL_PTP)) {
      sumkernel->analyze_dependencies();
      tmptasks = sumkernel->get_tasks();
      for (std::vector<task*>::iterator t=tmptasks->begin(); t!=tmptasks->end(); ++t)
	rtasks->push_back( *t );
    }

    this->set_kernel_tasks( rtasks );
  }
  void execute() {
    architecture *arch = get_out_object();
    //product_environment *env = (product_environment*)get_out_object()->get_environment();
    if (arch->has_collective_strategy(collective_strategy::GROUP)) {
      partial_summing->execute();
      top_summing->execute();
    } else if (arch->has_collective_strategy(collective_strategy::ALL_PTP)) {
      sumkernel->execute();
    }
  };
  //  int get_groupsize() { return groupsize; };
};

/*!
  An inner product is somewhat tricky. While it behaves as a single kernel,
  it is composed of two kernels: one gather followed by a local summing.
  This means we have to overwrite #analyze_dependencies and #execute
  to call the corresponding routines of the two enclosed kernels.

  \todo add an option for an ortho parameter
  \todo test that "global_sum" is replicated_scalar
  \todo test that v1 v2 have equal distributions
*/
#if 1
class product_innerproduct_kernel : public mpi_innerproduct_kernel,public product_kernel {
public:
  product_innerproduct_kernel( std::shared_ptr<object> v1,std::shared_ptr<object> v2,std::shared_ptr<object> global_sum)
    : kernel(v1,global_sum),product_kernel(v1,global_sum),
      mpi_innerproduct_kernel(v1,v2,global_sum),
      //innerproduct_kernel(v1,v2,global_sum),
      entity(entity_cookie::KERNEL) {};
};
#else
class product_innerproduct_kernel : virtual public product_kernel {
private:
  distribution *local_scalar; // we need to keep them just to destroy them
  std::shared_ptr<object> local_value;
protected:
  product_kernel *prekernel,*sumkernel;
public:
  product_innerproduct_kernel( std::shared_ptr<object> v1,std::shared_ptr<object> v2,std::shared_ptr<object> global_sum)
    : kernel(v1,global_sum),product_kernel(v1,global_sum) {
    architecture *arch = global_sum;
    //product_environment *env = (product_environment*)v1->get_environment();
    set_name("inner-product");

    // intermediate object for local sum:
    local_scalar = new product_block_distribution(arch,1,-1);
    local_value = new product_object(local_scalar);
    local_value->set_name("local-inprod-value");
    
    // local inner product kernel
    dependency *d;
    prekernel = new product_kernel(v1,local_value);
    prekernel->set_name("local-innerproduct");
    // v1 is in place
    d = prekernel->last_dependency();
    d->set_explicit_beta_distribution(v1);
    d->set_name("inprod wait for in vector");
    // v2 is in place
    prekernel->add_in_object(v2);
    d = prekernel->last_dependency();
    d->set_explicit_beta_distribution(v2);
    d->set_name("inprod wait for second vector");
    // function
    prekernel->set_localexecutefn( &local_inner_product );

    // reduction kernel
    sumkernel = new product_reduction_kernel(local_value,global_sum);
  };
  /* ~product_innerproduct() { delete local_scalar; */
  /*   delete local_value; delete prekernel; delete synckernel; delete sumkernel; }; */
  /*! We override the default analysis by analyzing in sequence the contained kernels.
    This splits the kernels and analyzes task dependencies; then we take the tasks
    and add them to the innerproduct task list.
   */
  virtual void analyze_dependencies() override {
    std::vector<task*> *rtasks = new std::vector<task*>, *tmptasks;

    prekernel->analyze_dependencies();
    sumkernel->analyze_dependencies();

    tmptasks = prekernel->get_tasks();
    for (std::vector<task*>::iterator t=tmptasks->begin(); t!=tmptasks->end(); ++t)
      rtasks->push_back( *t );
    tmptasks = sumkernel->get_tasks();
    for (std::vector<task*>::iterator t=tmptasks->begin(); t!=tmptasks->end(); ++t)
      rtasks->push_back( *t );
    this->set_kernel_tasks( rtasks );
  }
  //! We override the default execution by executing in sequence the contained kernels
  virtual void execute() { // synckernel->execute();
    prekernel->execute(); sumkernel->execute(); };
  kernel *get_prekernel() { return prekernel; };
};
#endif

/*!
  A vector norm is like an inner product, but since it has only one input
  we don't need the synckernel
*/
class product_norm_kernel : virtual public product_kernel {
private:
  distribution *local_scalar,*gathered_scalar;
  std::shared_ptr<object> local_value,*squared;
  int groupsize{4};
protected:
  product_kernel *prekernel,*sumkernel,*rootkernel;
public:
  product_norm_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out ) : kernel(in,out),product_kernel(in,out) {
    architecture *arch = out;
    //product_environment *env = (product_environment*)in->get_environment();
    set_name("norm");

    // intermediate object for local sum:
    local_scalar = new product_block_distribution(arch,1,-1);
    local_value = new product_object(local_scalar);
    local_value->set_name("local-inprod-value");
    
    // local norm kernel
    prekernel = new product_kernel(in,local_value);
    prekernel->set_name("local-norm");
    prekernel->last_dependency()->set_explicit_beta_distribution( in );
    prekernel->set_localexecutefn( &local_normsquared );

    // reduction kernel
    squared = new product_object(out->get_distribution());
    sumkernel = new product_reduction_kernel(local_value,squared);

    // we need to take a root
    rootkernel = new product_kernel(squared,out);
    rootkernel->set_name("root of squares");
    rootkernel->set_explicit_beta_distribution( squared );
    rootkernel->set_localexecutefn( &vectorroot );
  };
  //! We override the default analysis by analyzing in sequence the contained kernels.
  virtual void analyze_dependencies() override {
    std::vector<task*> *rtasks = new std::vector<task*>, *tmptasks;

    prekernel->analyze_dependencies();
    tmptasks = prekernel->get_tasks();
    for (std::vector<task*>::iterator t=tmptasks->begin(); t!=tmptasks->end(); ++t)
      rtasks->push_back( *t );

    sumkernel->analyze_dependencies();
    tmptasks = sumkernel->get_tasks();
    for (std::vector<task*>::iterator t=tmptasks->begin(); t!=tmptasks->end(); ++t)
      rtasks->push_back( *t );

    rootkernel->analyze_dependencies();
    tmptasks = rootkernel->get_tasks();
    for (std::vector<task*>::iterator t=tmptasks->begin(); t!=tmptasks->end(); ++t)
      rtasks->push_back( *t );

    this->set_kernel_tasks( rtasks );
  }
  //! We override the default execution by executing in sequence the contained kernels
  virtual void execute() {
    prekernel->execute();
    sumkernel->execute();
    rootkernel->execute();
  };
  int get_groupsize() { return groupsize; };
};

/*!
  Norm squared is much like norm. 
*/
class product_normsquared_kernel : virtual public product_kernel {
private:
  distribution *local_scalar,*gathered_scalar;
  std::shared_ptr<object> local_value;
protected:
  product_kernel *prekernel,*sumkernel;
public:
  product_normsquared_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out ) : kernel(in,out),product_kernel(in,out) {
    architecture *arch = out;
    //product_environment *env = (product_environment*)in->get_environment();
    set_name("norm squared");

    // intermediate object for local sum:
    local_scalar = new product_block_distribution(arch,1,-1);
    local_value = new product_object(local_scalar);
    local_value->set_name("local-inprod-value");
    
    // local norm kernel
    prekernel = new product_kernel(in,local_value);
    prekernel->set_name("local-norm");
    prekernel->last_dependency()->set_explicit_beta_distribution(in); 
    prekernel->set_localexecutefn( &local_normsquared );
    prekernel->set_localexecutectx( (void*)in );

    // reduction kernel
    sumkernel = new product_reduction_kernel(local_value,out);
  };
  //! We override the default analysis by analyzing in sequence the contained kernels.
  virtual void analyze_dependencies() override {
    std::vector<task*> *rtasks = new std::vector<task*>, *tmptasks;

    prekernel->analyze_dependencies();
    sumkernel->analyze_dependencies();

    tmptasks = prekernel->get_tasks();
    for (std::vector<task*>::iterator t=tmptasks->begin(); t!=tmptasks->end(); ++t)
      rtasks->push_back( *t );
    tmptasks = sumkernel->get_tasks();
    for (std::vector<task*>::iterator t=tmptasks->begin(); t!=tmptasks->end(); ++t)
      rtasks->push_back( *t );
    this->set_kernel_tasks( rtasks );
  }
  //! We override the default execution by executing in sequence the contained kernels
  virtual void execute() override { prekernel->execute(); sumkernel->execute(); };
};

/*!
  An outer product kernel is a data parallel application of a redundantly 
  distributed array of length k with a disjointly distributed array of 
  size N, giving Nk points. The redundant array is stored as context,
  and the explosion function is passed as parameter.
 */
class product_outerproduct_kernel : public product_kernel {
public:
 product_outerproduct_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out,
			  std::shared_ptr<object> replicated,
			  void (*f)(int, int, std::vector<object*>*, object*, void*)
			  ) : product_kernel(in,out) {
    set_name("outer product");
    localexecutefn = f;
    localexecutectx = (void*)replicated;
    dependency *d = last_dependency();
    d->set_type_local();
  };
  //! \todo why isn't this inherited from product_kernel
  task *make_task_for_domain(int d,std::shared_ptr<object> in,std::shared_ptr<object> out) {
    return new product_task(this->get_step(),d,in,out); };
};

#endif

class product_preconditioning_kernel : public product_kernel,public preconditioning_kernel {
public:
  product_preconditioning_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out )
    : kernel(in,out),product_kernel(in,out),preconditioning_kernel(in,out),
      entity(entity_cookie::KERNEL) {};
};

class product_trace_kernel : public product_kernel,public trace_kernel {
public:
  product_trace_kernel(std::shared_ptr<object> in,std::string c)
    : kernel(),trace_kernel(in,c),product_kernel(),
      entity(entity_cookie::KERNEL) {
    out_object = //in->object_with_same_distribution();
      std::shared_ptr<object>
          ( new product_object( dynamic_cast<distribution*>(in.get()) ) );
    out_object->set_name(fmt::format("trace-{}",in->get_name()));
  };
};

//! An MPI central differences kernel is exactly the same as \ref centraldifference_kernel
class product_centraldifference_kernel : public product_kernel,public centraldifference_kernel {
public:
  product_centraldifference_kernel(std::shared_ptr<object> in,std::shared_ptr<object> out)
    : kernel(in,out),product_kernel(in,out),centraldifference_kernel(in,out),
      entity(entity_cookie::KERNEL) {
  };
};

class product_diffusion_kernel : virtual public product_kernel {
public:
  product_diffusion_kernel(std::shared_ptr<object> in,std::shared_ptr<object> out)
    : kernel(in,out),product_kernel(in,out),
      entity(entity_cookie::KERNEL) {
    add_sigma_operator( ioperator("none") ); // we need the i index
    add_sigma_operator( ioperator(">=1") );  // we need the i+1 index
    add_sigma_operator( ioperator("<=1") );  // we need the i-1 index
    double damp = 1./6;
    set_localexecutefn
      ( [damp] (kernel_function_args) -> void {
	return central_difference_damp(kernel_function_call,damp); } );
  };
};

#endif
