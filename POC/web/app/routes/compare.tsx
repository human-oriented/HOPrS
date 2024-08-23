import { MetaFunction, ActionFunction, json, LinksFunction } from "@remix-run/node";
import { Form, Link, useFetcher } from "@remix-run/react";
import { useState, useEffect } from "react";
import * as Icon from 'react-feather';
import { jsonWithSuccess, jsonWithError } from "remix-toast";
import { ReactCompareSlider, ReactCompareSliderImage } from 'react-compare-slider';
import { LoadingButton } from "~/components/LoadingButton";
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import Switch from "react-switch";

export const meta: MetaFunction = () => {
    return [
      { title: "HOPrS" },
      { name: "description", content: "HOPrS - Human oriented proof standard" },
    ];
};

export const action: ActionFunction = async ({ request }) => {
    const formData = await request.formData();
    console.log('formadata', formData)
    const file = formData.get('file');
    const image = formData.get('image');
    const depth = formData.get('depth');
    const threshold = formData.get('threshold');
  
    if (!file || typeof file === 'string') {
      return json({ error: 'File is required' }, { status: 400 });
    }

    if (!image || typeof image === 'string') {
        return json({ error: 'Image is required' }, { status: 400 });
      }
  
    const externalApiUrl = `${process.env.API_URL}/hoprs/compare`; // Replace with your external API URL
    const externalFormData = new FormData();
    externalFormData.append('original_image_qt', file, file.name);
    externalFormData.append('new_image', image, image.name)
    externalFormData.append('compare_depth', depth?.toString() ?? '5');
    externalFormData.append('threshold', threshold?.toString() ?? '10');
    
    try {
      const response = await fetch(externalApiUrl, {
        method: 'POST',
        body: externalFormData,
      });
  
      if (!response.ok) {
        throw new Error('Failed to compare files, please check files and try again.');
      }
  
      const responseData = await response.json();
      console.log('response', responseData)
      return jsonWithSuccess({ success: true, data: responseData }, "Image comparison results ready!");
    } catch (error) {
        console.error(error)
      return jsonWithError({ error: (error as Error).message }, error?.message);
    }
  };

export default function Compare() {
    const fetcher = useFetcher()
    const [imageQT, setImageQT] = useState()
    const [image, setImage] = useState()
    const [threshold, setThreshold] = useState(2)
    const [depth, setDepth] = useState(5)
    const [loading, setLoading] = useState(false)
    const [checked, setChecked] = useState(false)

    useEffect(() => {
        if (fetcher?.state != "idle") {
            setLoading(true)
        } else {
            setLoading(false)
        }
    }, [fetcher?.state])

    const percent = (value) => {
        return Math.round(value * 100)
    }

    return (
    <>
    <div className="p-5 container m-auto text-center items-center justify-center flex flex-col flex-stretch min-h-screen pt-[120px] pb-[50px]">
      <div className="intro text-gray-500 lg:w-2/3 mx-auto mb-5 text-sm">
        <h1 className="font-bold text-3xl text-black">Image Comparison</h1>
        <p className="my-3">Using our tool you can compare an image to a quad tree file. The result of the comparison will show the similaritis between the files and will identify any edits that have been made to the image. </p>
        <p>Please use our <Link className="underline" to="/">encode</Link> tool to generate a quad tree file of any image you'd like to compare.</p>
      </div>
      <fetcher.Form method="POST" encType="multipart/form-data" className="flex flex-col w-full">
        <div className="files flex flex-row items-center">
            <div className="w-1/2 p-5">
                <div className="flex items-center justify-center w-full">
                    <label htmlFor="file" className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 dark:hover:bg-gray-800 dark:bg-gray-700 hover:bg-gray-100 dark:border-gray-600 dark:hover:border-gray-500 dark:hover:bg-gray-600 p-3">
                        <div className="flex flex-col items-center justify-center pt-5 pb-6 text-black w-full">
                            <Icon.FileText size={40} />
                            {imageQT ?
                            <>
                                <p className="text-sm text-gray-500 w-2/3 my-3">{imageQT?.name}</p>
                                <p className="bg-black px-5 py-2 text-white rounded-full text-sm">Change</p>
                            </>
                            :
                            <>
                                <p className="my-2 font-bold text-black">Quad tree file upload</p>
                                <p className="mb-2 text-sm text-gray-500 dark:text-gray-400"><span className="font-semibold">Click to upload</span> or drag and drop</p>
                                <p className="text-xs text-gray-500 dark:text-gray-400">.qt file (MAX. 500mb)</p>
                            </>
                            }
                        </div>
                        <input id="file" name="file" type="file" onChange={e => setImageQT(e?.target?.files[0])} className="sr-only" />
                    </label>
                </div> 
            </div>
            <div className="w-1/2 p-5">
                <div className="flex items-center justify-center w-full">
                    <label htmlFor="image" className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 dark:hover:bg-gray-800 dark:bg-gray-700 hover:bg-gray-100 dark:border-gray-600 dark:hover:border-gray-500 dark:hover:bg-gray-600 p-3">
                        <div className="flex flex-col items-center justify-center pt-5 pb-6 text-black w-full">
                            <Icon.Image size={40} />
                            {image ?
                            <>
                                <p className="text-sm text-gray-500 w-2/3 my-3">{image?.name}</p>
                                <p className="bg-black px-5 py-2 text-white rounded-full text-sm">Change</p>
                            </>
                            :
                            <>
                                <p className="my-2 font-bold text-black">Image upload</p>
                                <p className="mb-2 text-sm text-gray-500 dark:text-gray-400"><span className="font-semibold">Click to upload</span> or drag and drop</p>
                                <p className="text-xs text-gray-500 dark:text-gray-400">HEIC, PNG, JPG or GIF (MAX. 4032x3024px)</p>
                            </>
                            }
                        </div>
                        <input id="image" name="image" type="file" onChange={e => setImage(e?.target?.files[0])} className="sr-only" />
                    </label>
                </div> 
            </div>
        </div>
        
        <div className="options flex flex-col text-left items-center px-5">
            <div className="heading flex flex-row items-center">
                <button className="font-bold text-sm" type="button">Options</button>
                <Icon.ChevronDown size={15} />
            </div>
            
            <div className="options-inner hidden">
                <div className="flex flex-col w-1/2 text-left justify-start">
                    <label>Threshold</label>
                    <input type="number" name="threshold" defaultValue={threshold} />
                </div>
                <div className="flex flex-col w-1/2 items-center">
                    <label>Depth</label>
                    <input type="number" name="depth" defaultValue={depth} />
                </div>
            </div>
        </div>
        <div className="buttons my-10">
            <LoadingButton loading={loading} width={300} className="bg-black">Compare Image</LoadingButton>
        </div>
      </fetcher.Form>
      
      {fetcher?.data && fetcher?.data?.success && fetcher?.data?.data &&
        <div className="w-full h-full flex flex-col lg:flex-row items-center justify-center">
            <div className="lg:w-1/2 p-20 flex flex-col items-center justify-center order-1 lg:order-0">
                <p className="text-3xl font-bold mb-5">Image similarity</p>
                <CircularProgressbar 
                    className="w-full max-w-[180px]" 
                    value={percent(fetcher?.data?.data?.stats?.proportion)} 
                    text={`${percent(fetcher?.data?.data?.stats?.proportion)}%`}
                    styles={buildStyles({
                        // Rotation of path and trail, in number of turns (0-1)
                        rotation: 0,
                    
                        // Whether to use rounded or flat corners on the ends - can use 'butt' or 'round'
                        strokeLinecap: 'butt',
                    
                        // Text size
                        textSize: '16px',
                    
                        // How long animation takes to go from one percentage to another, in seconds
                        pathTransitionDuration: 0.5,
                    
                        // Can specify path transition in more detail, or remove it entirely
                        // pathTransition: 'none',
                    
                        // Colors
                        pathColor: `#000`,
                        textColor: '#000',
                        trailColor: '#ddd',
                        backgroundColor: '#eee',
                        
                      })} 
                />
                <div className="mt-5 text-left text-gray-400">
                    <p className="my-2 py-2 border-gray-200 border-b flex flex-row justify-between">Quad tree file: <span className="text-black ml-2">{imageQT?.name}</span></p>
                    <p className="my-2 py-2 border-gray-200 border-b flex flex-row justify-between">Image: <span className="text-black ml-2">{image?.name}</span></p>
                    <p className="my-2 py-2 border-gray-200 border-b flex flex-row justify-between">Threshold: <span className="text-black ml-2">{threshold}</span></p>
                    <p className="my-2 py-2 border-gray-200 border-b flex flex-row justify-between">Depth: <span className="text-black ml-2">{depth}</span></p>
                    <p className="my-2 py-2 border-gray-200 border-b flex flex-row justify-between">Total image pixels: <span className="text-black ml-2">{fetcher?.data?.data?.stats?.total_pixels}</span></p>
                    <p className="my-2 py-2 border-gray-200 border-b flex flex-row justify-between">Total matched pixels: <span className="text-black ml-2">{fetcher?.data?.data?.stats?.matched_pixels}</span></p>
                    <p className="my-2 py-2 flex flex-row justify-between">Perceptual algorithm: <span className="text-black ml-2">PDQ</span></p>
                </div>
            </div>
            <div className="w-full order-0 lg:order-1 lg:w-1/2 p-5 pt-0 lg:pt-5 flex flex-col items-center">
                <div className="flex flex-row items-center p-5 text-sm">
                    <p>Highlight image</p>
                    <Switch 
                        className="mx-3" 
                        onChange={() => setChecked(!checked)} 
                        checked={checked} 
                        checkedIcon={false} 
                        uncheckedIcon={false} 
                        offColor="#000"
                        onColor="#000"
                    />
                    <p>Quad tree image</p>
                </div>
                <ReactCompareSlider
                    position={50}
                    itemOne={<ReactCompareSliderImage src={fetcher.data.data.new_image} alt="Original image" />}
                    itemTwo={<ReactCompareSliderImage src={checked ? fetcher.data.data.comparison_image : fetcher.data.data.highlight_image} alt="Comparison image" />}
                />
            </div>
        </div>
      }

    </div>
    </>
  );
}
