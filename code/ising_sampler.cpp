// Replica-exchange Gibbs sampling for P(x) proportional to exp(h*x + sum_{i<j} J_ij*x_i*x_j).
// Each recorded state follows one complete sweep of the binary units at inverse temperature 1.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {
struct Random {
    uint64_t a, b;
    static uint64_t split(uint64_t& seed) {
        uint64_t x = (seed += 0x9e3779b97f4a7c15ULL);
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }
    explicit Random(uint64_t seed) : a(split(seed)), b(split(seed)) {}
    static uint64_t rot(uint64_t x, int n) { return (x << n) | (x >> (64-n)); }
    uint64_t next() {
        uint64_t x=a, y=b, result=rot(x+y,17)+x;
        y ^= x; a=rot(x,49)^y^(y<<21); b=rot(y,28);
        return result;
    }
    double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
};

struct State {
    std::array<double,64> field{};
    uint64_t bits=0;
    double energy=0;
};

void flip(State& state,int i,int n,const double* couplings) {
    uint64_t mask=uint64_t(1)<<i;
    double delta=(state.bits & mask) ? -1.0 : 1.0;
    state.energy+=delta*state.field[i];
    state.bits^=mask;
    const double* row=couplings+i*n;
    for(int k=0;k<n;k++) state.field[k]+=delta*row[k];
}
}

extern "C" int sample_network(int n,const double* h,const double* couplings,
        int ntemp,const double* beta,int burn,int draws,uint64_t seed,int start_dense,
        uint64_t* output,int64_t* accepted,int64_t* attempted) {
    if(n<1 || n>63 || ntemp<1 || burn<0 || draws<1 || beta[0]!=1.0) return 1;
    Random rng(seed);
    std::vector<State> pool(ntemp);
    std::vector<int> order(ntemp);
    for(int t=0;t<ntemp;t++) {
        order[t]=t;
        std::copy(h,h+n,pool[t].field.begin());
        for(int i=0;i<n;i++) if(start_dense || (t>0 && rng.uniform()<0.5)) flip(pool[t],i,n,couplings);
    }
    for(int sweep=0;sweep<burn+draws;sweep++) {
        for(int t=0;t<ntemp;t++) {
            State& state=pool[order[t]];
            // A systematic sweep leaves the target invariant; alternating direction removes a fixed scan preference.
            for(int k=0;k<n;k++) {
                int i=(sweep&1) ? (n-1-k) : k;
                bool old=(state.bits>>i)&1;
                double prob=1.0/(1.0+std::exp(-beta[t]*state.field[i]));
                if ((rng.uniform()<prob)!=old) flip(state,i,n,couplings);
            }
        }
        for(int t=sweep&1;t+1<ntemp;t+=2) {
            double ratio=(beta[t]-beta[t+1])*(pool[order[t+1]].energy-pool[order[t]].energy);
            if(sweep>=burn) attempted[t]++;
            if(ratio>=0 || rng.uniform()<std::exp(ratio)) {
                std::swap(order[t],order[t+1]);
                if(sweep>=burn) accepted[t]++;
            }
        }
        if(sweep>=burn) output[sweep-burn]=pool[order[0]].bits;
    }
    return 0;
}
