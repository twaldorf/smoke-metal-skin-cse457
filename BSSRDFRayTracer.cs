using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using CielaSpike;
using DefaultNamespace;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using static DefaultNamespace.MeshObject;

[RequireComponent(typeof(Camera))]
public class BSSRDFRayTracer : MonoBehaviour
{
    // Don't change these! Render the image as 512x512 for
    // the grade checking tool to work
    private const int Width = 512; 
    private const int Height = 512;
    private const int NTasks = 16; // number of parallel tasks to speed up raytracing
    public int MaxRecursionDepth = 3;

    public GameObject imageSavedText;
    public GameObject rawImage;
    public Camera renderCamera;
    public RenderTexture renderTexture;
    
    // You'll probably don't need to use these variables
    private CameraObject _cameraObject; // holds renderCamera data
    private List<MeshObject> _meshObjects; // stores all mesh objects in the scene
    private Color32[] _colors; // stores computed colors
    private Texture2D _tex2d; // texture that holds _colors
    private Ray _debugRay; // debug ray in Editor

    
    // You'll probably need to use these variables
    private BVH _bvh; // an instance of Bounding Volume Hierarchy acceleration structure, used to check for intersection
    private List<PointLightObject> _pointLightObjects; // point lights in the scene
    private Color _ambientColor; // ambient light in the scene
    private static readonly Color ReflectionRayColor = Color.blue;
    private static readonly Color RefractionRayColor = Color.yellow;
    private static readonly Color ShadowRayColor = Color.magenta;

    public Texture _SkyboxTexture;
    
    /// <summary>
    /// Initialize the necessary data and start tracing the scene
    /// (DO NOT MODIFY)
    /// </summary>
    public void Awake()
    {
        imageSavedText.SetActive(false);
        _colors = new Color32[Width * Height]; // holds ray-traced colors
        _tex2d = new Texture2D(Width, Height, TextureFormat.RGB24, false);
        
        _cameraObject = new CameraObject(Width, Height, renderCamera.cameraToWorldMatrix,
            Matrix4x4.Inverse(renderCamera.projectionMatrix), renderCamera.transform.position); // Initialize an instance of RenderCamera
        _meshObjects = CollectMeshes(); // Collect all meshes in the scene
        _pointLightObjects = CollectPointLights(); // Collect all point lights in the scene
        _ambientColor = RenderSettings.ambientLight; // Get the scene's ambient light
        _bvh = new BVH(_meshObjects); // Initialize an instance of accelerated ray-tracing structure
        
        StartCoroutine(TraceScene()); // Trace the scene
    }
    

    /// <summary>
    /// Trace the scene. We do this by tracing rays for each block of rows (TraceRows()) in parallel
    /// (DO NOT MODIFY)
    /// </summary>
    /// <returns></returns>
    private IEnumerator TraceScene()
    {
        List<Task> tasks = new List<Task>();
        
        var px = Width / NTasks;
        for (var i = 0; i < NTasks; i++) tasks.Add(new Task(TraceRows(0, Math.Min(px, Height))));
        // Initialize parallel ray tracing computations for each block of row
        for (var i = 0; i < NTasks; i++)
        {
            var startRow = i * px;
            var endRow = Math.Min((i + 1) * px, Height);
            Task task;
            this.StartCoroutineAsync(TraceRows(startRow, endRow), out task);
            tasks[i] = task;
        }

        for (var i = 0; i < NTasks; i++) yield return StartCoroutine(tasks[i].Wait());
        
        StartCoroutine(SaveTextureToFile()); // Save rendered image when complete
    }

    /// <summary>
    /// Trace rays from startRow to endRow
    /// (DO NOT MODIFY)
    /// </summary>
    /// <param name="startRow">the starting row</param>
    /// <param name="endRow">the ending row</param>
    /// <returns></returns>
    private IEnumerator TraceRows(int startRow, int endRow)
    {
        for (var i = startRow; i < endRow; i++)
        {
            for (var j = 0; j < Width; j++)
            {
                var ray = _cameraObject.ScreenToWorldRay(new Vector2(j, i));
                _colors[i * Width + j] = TraceRay(ray, 0, false, Color.red);
            }

            yield return Ninja.JumpToUnity;
            _tex2d.SetPixels32(_colors);
            _tex2d.Apply();
            rawImage.GetComponent<RawImage>().texture = _tex2d;
            yield return Ninja.JumpBack;
        }
    }

    // Shading utility functions
    private float FresnelReflectance(Vector3 H, Vector3 V, float Fnot) { 
        float zbase = 1.0f - Vector3.Dot(V, H);
        float exponential = (float)Math.Pow(zbase, 5.0);
        return (float)(exponential + Fnot * (1.0 - exponential)); 
    } 

    /// <summary>
    /// Trace a ray from the camera to a point on the screen and return the final color
    /// </summary>
    /// <param name="ray">a ray with origin and direction</param>
    /// <param name="recursionDepth">the current recursive level</param>
    /// <param name="debug">whether to draw the ray in the Editor</param>
    /// <param name="rayColor">the color (type) of the ray</param>
    /// <returns>the final color at a pixel</returns>
    private Color TraceRay(Ray ray, int recursionDepth, bool debug, Color rayColor)
    {

        Intersection hit;
        bool isHit = _bvh.IntersectBoundingBox(ray, out hit);  // IntersectBoundingBox checks for a potential intersection for a ray
    
        if (debug)    // Draw the rays
        {
            var hitPoint = ray.GetPoint(1000);
            if (isHit)
            {
                hitPoint = hit.point;
                Debug.DrawLine(hit.point, hit.point + (float)0.2 * hit.normal, Color.green);
            }
    
            Debug.DrawLine(ray.origin, hitPoint, rayColor);
        }
        
        if (!isHit) return Color.black; // Returns black when there's no intersection

        // An intersection occured, now get the necessary components
        var mat = hit.material;
        var kd = mat.Kd; // Diffuse component
        var ks = mat.Ks; // Specular component
        var ke = mat.Ke; // Emissive component
        var kt = mat.Kt; // Transparency component (refraction)
    
        var shininess = mat.Shininess;
        var indexOfRefraction = mat.IndexOfRefraction;
    
        var N = hit.normal;

        float attenuation = 0.0f;
        
        Color result = Color.black;

        result += ke;

        Color shade_kt = new Color(1,1,1,1);

        // (1) It's a good idea to check if the ray is entering or exiting an object...
        // TODO: Classify the medium, for BSSRDF

        bool entrance = Vector3.Dot(-N, ray.direction) > 0;
        if (!entrance) {
            N = -N;
            // In a medium, sample distance and eval transmission
            // Check distance to next surface
            // Scatter if distance is less than scatter distance of medium
            // Sample scatter direction
            // TraceRay
        }

        for (int i = 0; i < _pointLightObjects.Count; i++) {
            PointLightObject light = _pointLightObjects[i];

            Vector3 L = Vector3.Normalize(light.LightPos - hit.point);
            
            // shadow attenuation
            Ray to_light = new Ray(hit.point, L);
            Intersection shade_hit;
            bool isShadeHit = _bvh.IntersectBoundingBox(to_light, out shade_hit);

            while (isShadeHit) {
                if (
                    Vector3.Distance(hit.point, shade_hit.point) < Vector3.Distance(hit.point, light.LightPos)
                ) {
                    if (Vector3.Magnitude(new Vector3(shade_hit.material.Kt.r, shade_hit.material.Kt.g, shade_hit.material.Kt.b)) == 0) {
                        // this will stop when it finds a non-transparent object
                        shade_kt = new Color(0,0,0,0);
                        isShadeHit = false;
                    } else {
                        shade_kt *= shade_hit.material.Kt;
                        to_light = new Ray(shade_hit.point, L);
                        isShadeHit = _bvh.IntersectBoundingBox(to_light, out shade_hit);
                    }
                } else {
                    isShadeHit = false;
                }
            }

            Vector3 C_to_vtx = Vector3.Normalize(ray.origin - hit.point); // direction from camera to vertex
            Vector3 H = Vector3.Normalize(C_to_vtx + L); // halfway between camera and light direction
            float r = Vector3.Distance(hit.point, light.LightPos); // find r for attenuation

            // attenuate light
            attenuation = 1 / (float)((1 + Math.Pow(r, 2))); 

            // TODO: If in a medium, do not do this (it is replaced by internal transmission)

            // Diffuse component
            Color diffuse = new Color(0, 0, 0, 1);
            float diffuseShade = Math.Max(Vector3.Dot(N, L), 0.0f);
            diffuse = diffuseShade * kd * light.Color * light.Intensity * attenuation;

            // Specular component
            Color specular = new Color(0, 0, 0, 1);
            float specularShade = (float)Math.Max(Math.Pow(Vector3.Dot(N, H), shininess), 0.0f);
            specular = specularShade * ks * light.Color * light.Intensity * attenuation;

            // Surface reflection
            // Fnot from Beer's Law, assuming IOR of 1.4 (skin)
            float Fnot = 0.028f;
            float fresnel = FresnelReflectance(H, C_to_vtx, Fnot);

            // Add to sum with shadow attenuation
            result += shade_kt*(diffuse + specular);
        }

        // reflect and refract
        if (recursionDepth + 1 < MaxRecursionDepth) {
            Vector3 V = Vector3.Normalize(ray.origin - hit.point);

            // reflect
            if (Vector3.Magnitude(new Vector3(ks.r, ks.g, ks.b)) > 0f) {
                Vector3 reflection = 2 * Vector3.Dot(V, N) * N - V;
                Ray refl_ray = new Ray(hit.point, reflection);
                result += ks * TraceRay(refl_ray, recursionDepth + 1, debug, ReflectionRayColor);
            }

            // refract
            // TODO: Replace this with scattering event
            if (Vector3.Magnitude(new Vector3(kt.r, kt.g, kt.b)) > 0f) {
                float mew;
                if (entrance) {
                    mew = 1.0f / indexOfRefraction;
                } else {
                    mew = indexOfRefraction;
                }
                float cosThetaI = Vector3.Dot(N, V);
                float discriminant = (float)( 1 - ( Math.Pow(mew, 2) * (1 - Math.Pow(cosThetaI, 2)) ) );
                if (discriminant >= 0) { // check for total interal refraction
                    float cosThetaT = (float)Math.Sqrt(discriminant);
                    Vector3 T = (mew * cosThetaI - cosThetaT) * N - (mew * V);

                    Ray refr_ray = new Ray(hit.point, T);
                    Color refracted = TraceRay(refr_ray, recursionDepth + 1, debug, RefractionRayColor);

                    result += refracted * kt;
                }
            }
        }

        // Skybox ambient reflection sampler, TODO: Resolve get_isReadable() can only be called from the main thread error
        // int theta = Mathf.RoundToInt((float)(Math.Acos(ray.direction.y) / -Math.PI));
        // int phi = Mathf.RoundToInt((float)(Math.Atan2(ray.direction.x, -ray.direction.z) / -Math.PI * 0.5f));

        // Texture2D tmp = new Texture2D(RenderTexture.active.height, RenderTexture.active.width);
        // tmp.ReadPixels(new Rect(0,0,RenderTexture.active.height, RenderTexture.active.width), 0, 0);
        // tmp.Apply();
        // Color skybox = tmp.GetPixel(phi, theta);

        Color ambient = _ambientColor * kd;

        return result + ambient;
    }
    
    /// <summary>
    /// Draw a debug ray when user clicks somewhere in Game View
    /// (DO NOT MODIFY)
    /// </summary>
    public void Update()
    { if (Input.GetMouseButtonDown(0))
        {
            renderCamera.targetTexture = null;
            _debugRay = renderCamera.ScreenPointToRay(Input.mousePosition);
            renderCamera.targetTexture = renderTexture;
        }

        TraceRay(_debugRay,  0, true, Color.red);
    }
    

    /// <summary>
    /// Writes Texture2D to an image file which is used in the grade checking tool (ImageComparison.cs)
    /// (DO NOT MODIFY)
    /// </summary>
    private IEnumerator SaveTextureToFile()
    {
        var bytes = _tex2d.EncodeToPNG();
        var dirPath = Application.dataPath + "/Students/";
        Debug.Log("Rendered image saved to " + dirPath);
        if (!Directory.Exists(dirPath))
            Directory.CreateDirectory(dirPath);
        var mScene = SceneManager.GetActiveScene();
        var sceneName = mScene.name;
        File.WriteAllBytes(dirPath + sceneName + ".png", bytes);

        // Display "Image Saved" text
        imageSavedText.SetActive(true);
        yield return new WaitForSeconds(2);
        imageSavedText.SetActive(false);
    }

    /// <summary>
    /// Find and return all meshes in the scene
    /// (DO NOT MODIFY)
    /// </summary>
    /// <returns>A list of MeshObjects</returns>
    private List<MeshObject> CollectMeshes()
    {
        // Collect all meshes in the scene
        List<MeshObject> meshObjects = new List<MeshObject>();
        var meshRenderers = FindObjectsOfType<MeshRenderer>();

        foreach (var meshRenderer in meshRenderers)
        {
            var go = meshRenderer.gameObject;
            var mat = new Material(meshRenderer.material);
            var type = go.GetComponent<MeshFilter>().mesh.name == "Sphere Instance" ? "Sphere" : "TriMeshes";

            var sphereScale = go.transform.lossyScale;
            var sphereRadius = sphereScale.x / 2.0f; // A sphere so we only need to divide x by 2

            var m = go.GetComponent<MeshFilter>().mesh;
            var mo = new MeshObject(type, go, sphereRadius,
                go.transform.localToWorldMatrix, go.transform.position, mat,
                m.triangles, m.vertices, m.normals);
            meshObjects.Add(mo);
        }
        return meshObjects;
    }
    
    /// <summary>
    /// Find and return all point lights in the scene
    /// (DO NOT MODIFY)
    /// </summary>
    /// <returns>A list of PointLightObject</returns>
    private List<PointLightObject> CollectPointLights()
    {
        List<PointLightObject> lightObjects = new List<PointLightObject>();
        if (FindObjectsOfType(typeof(Light)) is Light[] lights)
        {
            for (var i = 0; i < lights.Length && lights[i].type == LightType.Point; i++)
            {
                var pos = lights[i].transform.position;
                var intensity = lights[i].intensity;
                var color = lights[i].color;
                lightObjects.Add(new PointLightObject(pos, intensity, color));
            }
        }
        return lightObjects;
    }
}