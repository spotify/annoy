/*
 * packed_index_test.cpp

 *
 *  Created on: Oct 3, 2018
 *      Author: Viper Craft
 *      Contact: viper.craft@gmail.com
 */

//#define __ERROR_PRINTER_OVERRIDE__(...) {}
#include <iostream>
#include "../src/kissrandom.h"
#include "../src/packedlib.h"
#include <random>
#include <vector>
#include <algorithm>

static char const TMP_FNAME[] = { "packed_annoy.idx" };

static float frand()
{
    return std::rand() / (float)RAND_MAX;
}

static std::vector<float> GenerateVector( size_t n, float lo, float hi )
{
    std::vector<float> v(n);
    std::transform
    (
        v.begin(), v.end(), v.begin(),
        [lo, hi]( float ) -> float { return lo + (hi - lo) * frand(); }
    );

    return v;
}

#define CHECK_AND_THROW(eq) { if( eq ) throw std::runtime_error("CHECK FAILED: " #eq); }

using namespace Annoy;

static int test(int f, int k, uint32_t count, int depth = 30)
{
    std::cout << "run test(), f=" << f << " k=" << k << " nvectors=" << count << std::endl;
    // create indexer first
    {
        PackedAnnoyIndexer<uint32_t, float, DotProduct, Kiss32Random> indexer(f, k);
        for( uint32_t i = 0; i < count; ++i )
        {
            auto vec = GenerateVector(f, -1.f, +1.f);
            indexer.add_item(i, vec.data());
        }
        std::cout << "build with depth=" << depth << " started." << std::endl;
        indexer.build(depth);
        std::cout << "building done, save into: \"" << TMP_FNAME << "\"" << std::endl;
        bool saved_success = indexer.save(TMP_FNAME);
        CHECK_AND_THROW(saved_success != true);
    }

    // and load from scratch

    PackedAnnoySearcher<uint32_t, float, DotProductPacked16> searcher;

    searcher.load(TMP_FNAME, false);

    uint32_t nitems = searcher.get_n_items(), nfound = 0;

    CHECK_AND_THROW( nitems != count );

    std::vector<uint32_t> results;

    size_t const search_k = (size_t)-1;

    uint32_t nitems_for_test = std::min(nitems, uint32_t(nitems * 0.5));

    std::cout << "scan start, nitems_for_test=" << nitems_for_test << std::endl;

    for( uint32_t i = 0; i < nitems_for_test; ++i )
    {
        // try to locate it in scan
        results.clear();
        searcher.get_nns_by_item(i, depth, search_k, &results, nullptr);
        if( std::find(results.begin(), results.end(), i) != results.end() )
            ++nfound;
    }

    double const qual = nfound / double(nitems_for_test);

    std::cout << "scan with depth=" << depth << " quality=" << qual << std::endl;

    return qual > 0.9 ? 0 : 1/*bad*/;
}


static int in_mem_test(int f, int k, uint32_t count, int depth = 30)
{
    std::cout << "run in_mem_test(), f=" << f << " k=" << k << " nvectors=" << count << std::endl;

    // make index into memory block

    detail::MMapWriter loader_n_writer;
    {
        PackedAnnoyIndexer<uint32_t, float, DotProduct, Kiss32Random> indexer(f, k);
        for( uint32_t i = 0; i < count; ++i )
        {
            auto vec = GenerateVector(f, -1.f, +1.f);
            indexer.add_item(i, vec.data());
        }
        std::cout << "build with depth=" << depth << " started." << std::endl;
        indexer.build(depth);
        std::cout << "building done, save into mmaped block ptr=" 
                  << loader_n_writer.get_ptr() << std::endl;
        // we can pass nullptr for filename
        bool saved_success = indexer.save_impl(loader_n_writer, nullptr);
        CHECK_AND_THROW(saved_success != true);
    }

    // and load from the same memory block

    PackedAnnoySearcher<uint32_t, float, DotProductPacked16, detail::MMapWriter> 
        searcher(std::move(loader_n_writer));

    // we can pass nullptr for filename
    searcher.load(nullptr, false);

    uint32_t nitems = searcher.get_n_items(), nfound = 0;

    CHECK_AND_THROW( nitems != count );

    std::vector<uint32_t> results;

    size_t const search_k = (size_t)-1;

    uint32_t nitems_for_test = std::min(nitems, uint32_t(nitems * 0.5));

    std::cout << "scan start, nitems_for_test=" << nitems_for_test << std::endl;

    for( uint32_t i = 0; i < nitems_for_test; ++i )
    {
        // try to locate it in scan
        results.clear();
        searcher.get_nns_by_item(i, depth, search_k, &results, nullptr);
        if( std::find(results.begin(), results.end(), i) != results.end() )
            ++nfound;
    }

    double const qual = nfound / double(nitems_for_test);

    std::cout << "scan with depth=" << depth << " quality=" << qual << std::endl;

    return qual > 0.9 ? 0 : 1/*bad*/;
}

static int run_tests()
{
    if( test(256, 256, 100000) )
        return 1;

    if( test(64, 128, 1000000) )
        return 1;

    if( in_mem_test(64, 128, 100000) )
        return 1;

    return 0;
}

int main(int argc, char **argv) {

    srand(336);
    try
    {
        if( run_tests() )
            throw std::runtime_error("some of the tests failed!");
    }
    catch(std::exception const &e)
    {
        std::cerr << e.what() << '\n';

        return EXIT_FAILURE;
    }


    std::cout << "SUCCESS\n";
    return EXIT_SUCCESS;
}
