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
#include <string>
#include <algorithm>

template<typename T>
bool is_near
(
    T const value,
    T const target,
    T const accuracy
)
{
    return
        (value + accuracy) >= target
        &&
        (value - accuracy) <= target
        ;
}

static char const TMP_FNAME[] = { "packed_annoy.idx" };

static float frand()
{
    return std::rand() / (float)RAND_MAX;
}

float vlength( std::vector<float> const &v )
{
    float sum = 0.f;

    for( float f : v )
        sum += f * f;

    return std::sqrt(sum);
}


float vlength( float const *v, uint32_t d )
{
    float sum = 0.f;

    for( uint32_t i = 0; i < d; ++i )
        sum += v[i] * v[i];

    return std::sqrt(sum);
}


void normalize( float const length, std::vector<float> &v )
{
    for( float &f : v )
        f /= length;
}

static std::vector<float> GenerateVectorNorm( size_t n, float lo, float hi )
{
    std::vector<float> v(n);
    float len;
    do
    {
        std::transform
        (
            v.begin(), v.end(), v.begin(),
            [lo, hi]( float ) -> float { return lo + (hi - lo) * frand(); }
        );

        // normalize vector for dot product!
        len = vlength(v);
    }
    while( is_near(len, 0.f, 0.00001f) );


    normalize(len, v);

    return v;
}

#define CHECK_AND_THROW(eq) \
    { if( !(eq) ) throw std::runtime_error("CHECK FAILED: " #eq " at line: " \
    + std::to_string(__LINE__)); }

#define COMPARE_FL(ev, val, acc) \
    { { auto real = ev; if( !is_near(real, val, acc) )\
    throw std::runtime_error("COMPARE FLOATS FAILED: " #ev " must be equal to: " #val \
    " but has: " + std::to_string(real) + " at line: " + std::to_string(__LINE__)); } }

using namespace Annoy;


static float test_vector[512] __attribute__ ((aligned (16))) = {0.04346,-0.009254,-0.05142,-0.01738,-0.03543,0.072,0.02031,0.03525,0.01837,0.001751,-0.03854,-0.02718,-0.02945,-0.00668,0.07086,-0.06445,0.02716,0.01264,
-0.01654,-0.04462,-0.03827,0.0526,0.0094,-0.01229,0.0198,0.04608,-0.001211,0.08453,-0.02464,0.001965,-0.01262,-0.01982,-0.01321,0.03143,0.01997,-0.04095,0.014565,0.06714,0.0169,0.02919,-0.0929,-0.009575,0.08453,-0.03314,-0.006638,-0.02513,-0.01967,-0.02611,0.002512,-0.0616,0.05316,-0.01212,-0.0695,0.10876,-0.01569,-0.04214,-0.02525,-0.05493,-0.10583,-0.02792,-0.08685,-0.03015,-0.003382,-0.0301,-0.01819,-0.03123,-0.025,-0.03065,0.014404,0.0355,-0.0901,0.10345,0.01227,0.002533,-0.01773,-0.02637,0.1107,0.003601,-0.0676,0.004215,-0.03268,0.04907,-0.005127,-0.06207,0.02179,0.01461,-0.02954,0.01548,-0.11084,-0.02773,0.09247,0.01374,-0.08185,-0.02985,-0.0379,-0.01451,-0.01457,-0.0792,-0.0657,0.0864,-0.007996,0.03363,0.02112,0.0043,0.06396,-0.0754,-0.0644,0.01339,-0.0502,-0.002678,-0.04056,-0.01521,0.02037,0.00841,0.03726,-0.0587,-0.0001751,-0.012665,-0.05154,-0.01834,0.004406,0.01279,-0.03488,0.01263,-0.0522,0.0696,
0.0371,0.011635,-0.01956,0.0909,0.00923,0.01108,0.01314,-0.07544,-0.003347,0.02063,-0.0112,-0.1076,0.00999,
-0.011055,0.01528,-0.05182,-0.06055,0.007072,-0.05438,-0.0666,-0.003635,0.01453,-0.006462,-0.0389,0.01071,0.0725,0.03955,0.002186,0.03108,-0.02351,0.0384,
0.0349,0.09076,0.0685,0.03485,0.02748,-0.00874,-0.0412,-0.0074,0.1021,0.01985,0.0017605,0.00393,0.03134,-0.04703,-0.00654,0.1228,0.0491,-0.06,-0.05978,
0.03882,-0.003521,-0.01717,0.0471,0.0859,0.004837,0.0528,0.0446,0.08795,-0.06335,0.007313,0.01359,0.005512,-0.00715,-0.01627,-0.01854,0.02142,0.01639,
0.08875,0.04282,-0.02362,0.07904,0.03017,0.0213,-0.02805,-0.0003293,0.0197,-0.07196,0.001445,-0.0315,0.04553,-0.03793,0.02069,0.01188,0.1027,-0.006516,
0.03497,-0.01483,0.06104,-0.03793,0.03436,-0.0459,-0.03555,-0.01991,-0.01228,0.10925,0.05426,-0.0128,-0.00836,-0.02063,-0.04355,-0.0327,0.02425,0.0621,
-0.01234,-1.63e-05,-0.01562,-0.004116,-0.00944,-0.006325,-0.004116,-0.02528,0.02719,-0.04968,-0.01204,0.0318,0.00844,-0.01878,-0.06885,0.003998,-0.00385,
0.01117,0.01674,-0.004135,0.03793,-0.03174,0.02661,0.04193,-0.05908,0.0284,0.05603,-0.0608,-0.0466,0.014496,-0.0355,-0.01522,-0.05484,0.06,-0.0397,-0.0691,
-0.003729,0.07117,0.02942,-0.04358,-0.1346,0.0472,-0.01843,0.005444,-0.03915,0.0864,-0.00384,0.02104,-0.003803,0.01473,-0.0333,-0.000556,0.01248,-0.0668,
0.02763,0.01397,0.002903,-0.02954,-0.01397,0.05026,-0.001897,0.0192,0.02708,0.04092,-0.06335,0.03687,-0.02068,0.02272,-0.05798,-0.014275,0.02295,-0.001261,
-0.06714,0.0437,-0.005787,0.05402,-0.02962,-0.02121,0.08386,0.01537,-0.04803,0.002726,0.03748,0.02583,0.0508,-0.04956,-0.02051,0.01289,-0.04156,0.01251,
-0.003637,0.0541,0.01866,0.066,0.02473,-0.013054,0.00833,0.02083,-0.004128,0.0599,0.04077,0.004692,-0.01359,0.0259,0.001337,0.03546,-0.003096,0.0685,
-0.0196,0.05045,-0.01223,-0.03894,-0.02191,-0.00635,0.02213,0.011925,-0.04727,-0.0628,-0.02505,0.01044,-0.0215,-0.0453,0.05945,-0.01746,-0.0701,0.03204,
-0.03223,0.0615,0.0095,0.0702,-0.0209,-0.02022,0.0671,-0.03387,-0.01411,0.1141,0.0653,-0.04648,0.0854,-0.06476,-0.03503,-0.001089,-0.07495,-0.01823,
-0.007103,0.0477,-0.01451,0.004215,0.01991,-0.02652,-0.01866,0.0516,-0.04236,0.02255,-0.056,0.02951,-0.07477,0.01884,0.0813,-0.0884,-0.0083,-0.0711,
-0.0004997,0.01179,-0.06396,-0.0001268,-0.04443,0.0966,0.02739,-0.05518,-0.05045,0.0481,-0.04507,0.07904,-0.0533,-0.004543,-0.004692,0.04385,-0.03702,0.005356,0.03384,-0.03214,0.04404,-0.03415,0.08734,-0.00944,-0.001612,-0.03345,
-0.02525,-0.02405,0.1087,-0.012985,0.0291,-0.0093,0.082,-0.04208,-0.09216,0.01566,-0.0412,-0.06177,0.06052,-0.0002166,0.00834,0.0673,-0.007046,-0.0194,-0.09906,-0.0389,0.00807,-0.02905,-0.06238,0.00964,0.02727,-0.015594,-0.0601,0.0375,0.02965,-0.04276,-0.0551,0.01419,-0.07465,-0.014145,0.015,0.04318,-0.0202,-0.03674,0.006294,0.02702,-0.0283,0.09674,-0.01744,-0.05453,-0.04523,-0.01168,-0.0519,-9.483e-05,0.04205,-0.00902,-0.02861,0.0523,-0.03842,-0.02223,-0.02263,-0.0496,-0.007664,-0.009415,0.06155,-0.006504,0.004177,-0.09546,-0.03427,-0.01443,0.00944,-0.02184,0.0813,-0.04834,0.04755,0.0006766,-0.0408,-0.0748,-0.0611,-0.01397,0.0574,-0.06976,0.05396,0.02792,-0.03717,-0.01231,-0.08496,-0.026,0.01575,0.03586,-0.01549,-0.0668,-0.01634,0.05606,-0.06915,-0.0701,-0.03067,-0.01309,-0.0008216,-0.07513
};

uint32_t search_with_filtering(PackedAnnoySearcher<uint32_t, float, DotProductPacked16> &searcher, int depth, uint32_t nitems_for_test )
{
    size_t const search_k = (size_t)-1;
    uint32_t nfound = 0;
    std::vector<std::pair<float, uint32_t>> results;
    for( uint32_t i = 0; i < nitems_for_test; ++i )
    {
        results.clear();
        searcher.get_nns_by_item_filter(i, depth, search_k, []( float &dist ) {
            // for DotProduct we must get abs to get real distance similarity
            dist = std::abs(dist);
            // also check for minimal quality
            if( dist > 0.8f )
                return true;
            return false; // throw away this result!
        }, results);
        // check results
        for( auto p : results )
        {
            // check all distances must be geq 0.1
            CHECK_AND_THROW( p.first >= 0.8f );
            if( i == p.second )
            {
                CHECK_AND_THROW( is_near(p.first, 1.f, 0.0001f) );
                ++nfound;
            }
        }
    }

    return nfound;
}

uint32_t search_with_filtering(PackedAnnoySearcher<uint32_t, float, EuclideanPacked16> &searcher, int depth, uint32_t nitems_for_test )
{
    size_t const search_k = (size_t)-1;
    uint32_t nfound = 0;
    std::vector<std::pair<float, uint32_t>> results;
    float const max_dist = 0.6f;
    for( uint32_t i = 0; i < nitems_for_test; ++i )
    {
        results.clear();
        searcher.get_nns_by_item_filter(i, depth, search_k, [max_dsqr = max_dist * max_dist]( float &dist ) {
            // for Euclidean we have squared distances here, so bound must be
            // squared before comparions!
            // also check for minimal quality
            if( dist < max_dsqr )
            {
                // distance passed so we can normalize it here!
                dist = EuclideanPacked16::normalized_distance(dist);
                return true;
            }
            return false; // throw away this result!
        }, results);
        // check results
        for( auto p : results )
        {
            // check all distances must be leq 0.6
            CHECK_AND_THROW( p.first < 0.6f );
            if( i == p.second )
            {
                CHECK_AND_THROW( is_near(p.first, 0.f, 0.0001f) );
                ++nfound;
            }
        }
    }

    return nfound;
}

static constexpr float _get_distance_bound( EuclideanPacked16 )
{
    return 0.f;
}

static constexpr float _get_distance_bound( DotProductPacked16 )
{
    return 1.f;
}

template<typename D, typename M>
void check_self_distances(PackedAnnoySearcher<uint32_t, float, D, M> &searcher, uint32_t nitems_for_test )
{

    std::cout << "check self distances start. " << std::endl;
    uint32_t nerrors = 0;
    double avg_dist = 0., avg_err_dist = 0.;
    for( uint32_t i = 0; i < nitems_for_test; ++i )
    {
        float dist = searcher.get_distance(i, i);
        if( !is_near(dist, _get_distance_bound(D{}), 0.0001f) )
        {
            ++nerrors;
            avg_err_dist += dist;
        }

        avg_dist += dist;
    }

    if( nerrors )
        std::cout << "found nerrors " << nerrors << " nitems "
                  << nitems_for_test << " avg_dist " << (avg_dist / nitems_for_test)
                  << " avg_dist_err " << (avg_err_dist / nerrors)
                  << std::endl;

    CHECK_AND_THROW( nerrors == 0 );
}

template<typename DistT>
static void test(int f, int k, uint32_t count, int depth = 30)
{
    // reset srand every time to get same vectors
    srand(336);
    std::cout << "run test() for " << typeid(DistT).name()
              << ", f=" << f << " k=" << k << " nvectors=" << count << std::endl;
    // create indexer first
    {
        PackedAnnoyIndexer<uint32_t, float, typename DistT::UnpackedT, Kiss32Random> indexer(f, k);
        indexer.verbose(true);
        for( uint32_t i = 0; i < count; ++i )
        {
            auto vec = GenerateVectorNorm(f, -1.f, +1.f);
            indexer.add_item(i, vec.data());
        }
        std::cout << "build with depth=" << depth << " started." << std::endl;
        indexer.build(depth);
        std::cout << "building done, save into: \"" << TMP_FNAME << "\"" << std::endl;
        bool saved_success = indexer.save(TMP_FNAME);
        CHECK_AND_THROW(saved_success == true);
    }

    // and load from scratch

    PackedAnnoySearcher<uint32_t, float, DistT> searcher;

    searcher.load(TMP_FNAME, false);

    uint32_t nitems = searcher.get_n_items(), nfound = 0;

    CHECK_AND_THROW( nitems == count );

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

    double qual = nfound / double(nitems_for_test);

    std::cout << "scan self with depth=" << depth << " quality=" << qual << std::endl;

    CHECK_AND_THROW( qual > 0.9 );

    // check with filtering
    nfound = search_with_filtering(searcher, depth, nitems_for_test);
    qual = nfound / double(nitems_for_test);

    std::cout << "scan vectors w/ filtering, with depth=" << depth << " quality=" << qual << std::endl;

    CHECK_AND_THROW( qual > 0.9 );


    check_self_distances(searcher, nitems);
}

template<typename S>
static double generic_selftest( S *searcher, uint32_t count, int depth )
{
    uint32_t nitems = searcher->get_n_items(), nfound = 0;

    CHECK_AND_THROW( nitems == count );

    std::vector<uint32_t> results;

    size_t const search_k = (size_t)-1;

    uint32_t nitems_for_test = std::min(nitems, uint32_t(nitems * 0.5));
    std::cout << "scan start, nitems_for_test=" << nitems_for_test << std::endl;

    for( uint32_t i = 0; i < nitems_for_test; ++i )
    {
        // try to locate it in scan
        results.clear();
        searcher->get_nns_by_item(i, depth, search_k, &results, nullptr);
        if( std::find(results.begin(), results.end(), i) != results.end() )
            ++nfound;
    }

    double const qual = nitems_for_test ? nfound / double(nitems_for_test) : 0.;

    std::cout << "scan with depth=" << depth << " quality=" << qual << std::endl;

    return qual;
}


template<typename DistT>
static double in_mem_test(int f, int k, uint32_t count, int depth = 30, bool clone = false)
{
    std::cout << "run in_mem_test(), f=" << f << " k=" << k << " nvectors=" << count << std::endl;

    // make index into memory block

    detail::MMapWriter loader_n_writer;
    {
        PackedAnnoyIndexer<uint32_t, float, typename DistT::UnpackedT, Kiss32Random> indexer(f, k);
        indexer.verbose(true);
        for( uint32_t i = 0; i < count; ++i )
        {
            auto vec = GenerateVectorNorm(f, -1.f, +1.f);
            indexer.add_item(i, vec.data());
        }
        std::cout << "build with depth=" << depth << " started." << std::endl;
        indexer.build(depth);
        // we can pass nullptr for filename
        bool saved_success = indexer.save_impl(loader_n_writer, nullptr);
        std::cout << "building done, save into mmaped block ptr="
                  << loader_n_writer.get_ptr() << std::endl;
        CHECK_AND_THROW(saved_success == true);
    }

    // and load from the same memory block
    using Searcher = PackedAnnoySearcher<uint32_t, float, DistT, detail::MMapWriter>;

    Searcher searcher(std::move(loader_n_writer));

    // we can pass nullptr for filename
    searcher.load(nullptr, false);

    double qual1 = generic_selftest(&searcher, count, depth);

    if( clone )
    {
        std::unique_ptr<Searcher> c( searcher.clone() );

        double qual2 = generic_selftest(c.get(), count, depth);

        CHECK_AND_THROW( qual1 == qual2 );

        // and clone of clone
        {
            std::unique_ptr<Searcher> c2( c->clone() );
            double qual3 = generic_selftest(c2.get(), count, depth);

            CHECK_AND_THROW( qual1 == qual3 );
        }
    }

    check_self_distances(searcher, count);

    return qual1;
}


void basic_packutils_test()
{
    COMPARE_FL( euclidean_distance(test_vector, test_vector, 512), 0.f, 0.00001f );
    __attribute__ ((aligned (16))) uint16_t packed[512] = {0};
    __attribute__ ((aligned (16))) float unpacked[512] = {0};
    pack_float_vector_i16(test_vector, packed, 512);
    decode_vector_i16_f32(packed, unpacked, 512);
    COMPARE_FL( decode_and_euclidean_distance_i16_f32(packed, test_vector, 512), 0.f, 0.00001f );
    COMPARE_FL( decode_and_dot_i16_f32(packed, test_vector, 512), 1.f, 0.0002f );
    std::vector<float> tvv(test_vector, test_vector + 512);
    float vlen = vlength(tvv);
    normalize(vlen, tvv);
    COMPARE_FL( vlength(tvv), 1.f, 0.0001f );
    COMPARE_FL( vlength(unpacked, 512), 1.f, 0.0001f );

    // binary compatible test across all packing/unpacking methods

#if defined(USE_AVX512)
    __attribute__ ((aligned (16))) uint16_t packed_avx32[512] = {0};
    __attribute__ ((aligned (16))) float unpacked_avx32[512] = {0};
    pack_float_vector_i16_avx32(test_vector, packed_avx32, 512);
    CHECK_AND_THROW( memcmp(packed, packed_avx32, sizeof(packed)) == 0 );

    COMPARE_FL( decode_and_euclidean_distance_i16_f32_avx32(packed, test_vector, 512), 0.f, 0.00001f );
    COMPARE_FL( decode_and_euclidean_distance_i16_f32_avx32(packed_avx32, test_vector, 512), 0.f, 0.00001f );
    COMPARE_FL( decode_and_euclidean_distance_i16_f32_sse(packed_avx32, test_vector, 512), 0.f, 0.00001f );
    COMPARE_FL( decode_and_dot_i16_f32_avx32(packed, test_vector, 512), 1.f, 0.0002f );

    decode_vector_i16_f32_avx32(packed, unpacked_avx32, 512);
    CHECK_AND_THROW( memcmp(unpacked, unpacked_avx32, sizeof(packed)) == 0 );
#endif
#if defined(USE_AVX2)
    __attribute__ ((aligned (16))) uint16_t packed_avx16[512] = {0};
    __attribute__ ((aligned (16))) float unpacked_avx16[512] = {0};
    pack_float_vector_i16_avx16(test_vector, packed_avx16, 512);
    CHECK_AND_THROW( memcmp(packed, packed_avx16, sizeof(packed)) == 0 );

    COMPARE_FL( decode_and_euclidean_distance_i16_f32_avx16(packed, test_vector, 512), 0.f, 0.00001f );
    COMPARE_FL( decode_and_euclidean_distance_i16_f32_sse(packed_avx16, test_vector, 512), 0.f, 0.00001f );
    COMPARE_FL( decode_and_dot_i16_f32_avx16(packed, test_vector, 512), 1.f, 0.0002f );


    decode_vector_i16_f32_avx16(packed, unpacked_avx16, 512);
    CHECK_AND_THROW( memcmp(unpacked, unpacked_avx16, sizeof(packed)) == 0 );
#endif

    COMPARE_FL( decode_and_euclidean_distance_i16_f32_sse(packed, test_vector, 512), 0.f, 0.00001f );
    COMPARE_FL( decode_and_dot_i16_f32_sse(packed, test_vector, 512), 1.f, 0.0002f );
}

int main(int argc, char **argv) {
    try
    {
        // basic utils
        basic_packutils_test();
        // DotProduct
        test<DotProductPacked16>(256, 256, 100000);
        test<DotProductPacked16>(64, 64, 1000000);
        // and hard case for avx, causes a split
        test<DotProductPacked16>(40, 40, 100000);
        // Euclidean
        test<EuclideanPacked16>(256, 256, 100000);
        test<EuclideanPacked16>(64, 64, 1000000);
        // and hard case for avx, causes a split
        test<EuclideanPacked16>(40, 40, 100000);
        // test in mem and clones
        CHECK_AND_THROW( in_mem_test<DotProductPacked16>(64, 64, 10000, 30, true) > 0.9 );
        CHECK_AND_THROW( in_mem_test<EuclideanPacked16>(64, 64, 10000, 30, true) > 0.9 );
        // in the case we try to make very small index
        CHECK_AND_THROW( in_mem_test<DotProductPacked16>(64, 64, 17) >= 0.25 );
        // edge cases
        for( int v_sz : { 64, 128, 256, 512 } )
            for( int i_sz : { 16, 32, 64, 128, 512 } )
                for( int c_sz : { 0, 1, 3, 17, 33, 200 } )
                    for( int depth : { 30, 50, 100, 200 } )
                        if( v_sz >= i_sz )
                            CHECK_AND_THROW( in_mem_test<DotProductPacked16>(v_sz, i_sz, c_sz, depth) >= 0 );
    }
    catch(std::exception const &e)
    {
        std::cerr << e.what() << '\n';

        return EXIT_FAILURE;
    }


    std::cout << "SUCCESS\n";
    return EXIT_SUCCESS;
}
